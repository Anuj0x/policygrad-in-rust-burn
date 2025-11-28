//! Training utilities and configuration for policy gradient algorithms

use burn::optim::{Adam, AdamConfig, Optimizer};
use burn::tensor::{backend::Backend, Tensor};
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Hyperparameter configuration for algorithms
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    pub learning_rate: f64,
    pub max_grad_norm: f32,
    pub batch_size: usize,
    pub num_minibatches: usize,
    pub update_epochs: usize,
    pub total_timesteps: usize,
    pub anneal_lr: bool,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_eps: f32,
    pub vf_cost: f32,
    pub entropy_cost: f32,
    pub target_kl: Option<f32>,
    pub normalize_advantages: bool,
}

/// Environment configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvConfig {
    pub env_id: String,
    pub num_envs: usize,
    pub normalize_observations: bool,
    pub normalize_rewards: bool,
    pub clip_observations: f32,
    pub clip_rewards: f32,
}

/// Network architecture configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub policy_hidden_sizes: Vec<usize>,
    pub value_hidden_sizes: Vec<usize>,
    pub activation: String,
    pub squash_distribution: bool,
}

/// Training configuration combining all components
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub algorithm: AlgorithmConfig,
    pub environment: EnvConfig,
    pub network: NetworkConfig,
    pub experiment_name: String,
    pub seed: u64,
    pub save_model: bool,
    pub eval_every: usize,
    pub num_eval_episodes: usize,
    pub deterministic_eval: bool,
}

/// Training statistics and metrics
#[derive(Clone, Debug, Default)]
pub struct TrainingMetrics {
    pub total_steps: usize,
    pub updates: usize,
    pub sps: f32,
    pub walltime: f32,
    pub rollout_time: f32,
    pub update_time: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
    pub clip_fraction: f32,
}

/// Evaluation metrics
#[derive(Clone, Debug)]
pub struct EvaluationMetrics {
    pub num_episodes: usize,
    pub num_steps: usize,
    pub mean_score: f32,
    pub std_score: f32,
    pub mean_episode_length: f32,
    pub std_episode_length: f32,
    pub eval_time: f32,
}

/// Generic trainer for policy gradient algorithms
pub struct Trainer<B: Backend, A> {
    config: TrainingConfig,
    algorithm: A,
    optimizer: Adam<B>,
    start_time: Instant,
    global_step: usize,
}

impl<B: Backend, A> Trainer<B, A> {
    pub fn new(config: TrainingConfig, algorithm: A) -> Self {
        let optimizer = AdamConfig::new()
            .with_lr(config.algorithm.learning_rate)
            .init();

        Self {
            config,
            algorithm,
            optimizer,
            start_time: Instant::now(),
            global_step: 0,
        }
    }

    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    pub fn global_step(&self) -> usize {
        self.global_step
    }

    pub fn update_global_step(&mut self, steps: usize) {
        self.global_step += steps;
    }

    pub fn optimizer(&mut self) -> &mut Adam<B> {
        &mut self.optimizer
    }

    pub fn algorithm(&self) -> &A {
        &self.algorithm
    }

    pub fn algorithm_mut(&mut self) -> &mut A {
        &mut self.algorithm
    }

    pub fn start_time(&self) -> Instant {
        self.start_time
    }

    pub fn walltime(&self) -> f32 {
        self.start_time.elapsed().as_secs_f32()
    }
}

/// Learning rate scheduler
pub struct LRScheduler {
    initial_lr: f64,
    final_lr: f64,
    total_steps: usize,
    current_step: usize,
}

impl LRScheduler {
    pub fn new(initial_lr: f64, final_lr: f64, total_steps: usize) -> Self {
        Self {
            initial_lr,
            final_lr,
            total_steps,
            current_step: 0,
        }
    }

    pub fn step(&mut self) -> f64 {
        if self.current_step >= self.total_steps {
            return self.final_lr;
        }

        let progress = self.current_step as f64 / self.total_steps as f64;
        let lr = self.initial_lr + progress * (self.final_lr - self.initial_lr);

        self.current_step += 1;
        lr
    }

    pub fn current_lr(&self) -> f64 {
        if self.current_step >= self.total_steps {
            return self.final_lr;
        }

        let progress = self.current_step as f64 / self.total_steps as f64;
        self.initial_lr + progress * (self.final_lr - self.initial_lr)
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Utility functions for training
pub struct TrainingUtils;

impl TrainingUtils {
    /// Compute generalized advantage estimation (GAE)
    pub fn compute_gae<B: Backend>(
        rewards: &Tensor<B, 2>,
        values: &Tensor<B, 2>,
        next_values: &Tensor<B, 2>,
        dones: &Tensor<B, 2>,
        gamma: f32,
        gae_lambda: f32,
    ) -> Tensor<B, 2> {
        let mut advantages = Tensor::<B, 2>::zeros_like(rewards);

        let mut last_gae_lam = Tensor::<B, 2>::zeros_like(next_values.select(0, 0));

        for step in (0..rewards.dims()[0]).rev() {
            let reward = rewards.select(0, step);
            let value = values.select(0, step);
            let next_value = if step == rewards.dims()[0] - 1 {
                next_values.select(0, 0)
            } else {
                values.select(0, step + 1)
            };
            let done = dones.select(0, step);

            let delta = reward + gamma * next_value * (1.0 - done) - value;
            last_gae_lam = delta + gamma * gae_lambda * (1.0 - done) * last_gae_lam;

            advantages = advantages.index_assign([step..step+1, 0..], last_gae_lam.unsqueeze_dim(0));
        }

        advantages
    }

    /// Normalize advantages
    pub fn normalize_advantages<B: Backend>(advantages: &Tensor<B, 2>) -> Tensor<B, 2> {
        let mean = advantages.mean();
        let std = advantages.std(0) + 1e-8;
        (advantages - mean) / std
    }

    /// Compute returns using discounted rewards
    pub fn compute_returns<B: Backend>(
        rewards: &Tensor<B, 2>,
        dones: &Tensor<B, 2>,
        gamma: f32,
    ) -> Tensor<B, 2> {
        let mut returns = Tensor::<B, 2>::zeros_like(rewards);

        let mut running_return = Tensor::<B, 2>::zeros_like(rewards.select(0, 0));

        for step in (0..rewards.dims()[0]).rev() {
            let reward = rewards.select(0, step);
            let done = dones.select(0, step);

            running_return = reward + gamma * running_return * (1.0 - done);
            returns = returns.index_assign([step..step+1, 0..], running_return.unsqueeze_dim(0));
        }

        returns
    }

    /// Create minibatches from rollout data
    pub fn create_minibatches<B: Backend>(
        data: &[Tensor<B, 2>],
        batch_size: usize,
        num_minibatches: usize,
    ) -> Vec<Vec<Tensor<B, 2>>> {
        let total_samples = data[0].dims()[0];
        let samples_per_minibatch = batch_size / num_minibatches;

        (0..num_minibatches)
            .map(|_| {
                // Random permutation
                let indices = Tensor::<B, 1>::arange(0..total_samples as i64, &Default::default())
                    .shuffle(&mut rand::thread_rng());

                data.iter()
                    .map(|tensor| {
                        indices
                            .narrow(0, 0, samples_per_minibatch)
                            .select_from_tensor(tensor)
                    })
                    .collect()
            })
            .collect()
    }

    /// Log training metrics
    pub fn log_metrics(metrics: &TrainingMetrics, step: usize) {
        tracing::info!(
            "Step: {}, Loss: {:.4}, Policy: {:.4}, Value: {:.4}, Entropy: {:.4}, KL: {:.4}, SPS: {:.1}",
            step,
            metrics.policy_loss + metrics.value_loss + metrics.entropy_loss,
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy_loss,
            metrics.approx_kl,
            metrics.sps
        );
    }

    /// Save model checkpoint
    pub fn save_checkpoint<B: Backend, M: burn::module::Module<B>>(
        model: &M,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(path.parent().unwrap())?;
        let file = std::fs::File::create(path)?;
        burn::record::BinFileRecorder::new().record(model, file)?;
        Ok(())
    }

    /// Load model checkpoint
    pub fn load_checkpoint<B: Backend, M: burn::module::Module<B>>(
        model: &mut M,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        burn::record::BinFileRecorder::new().load(model, file)?;
        Ok(())
    }
}

/// Default configurations for different algorithms
pub struct ConfigFactory;

impl ConfigFactory {
    pub fn ppo_config() -> TrainingConfig {
        TrainingConfig {
            algorithm: AlgorithmConfig {
                learning_rate: 3e-4,
                max_grad_norm: 0.5,
                batch_size: 1,
                num_minibatches: 8,
                update_epochs: 10,
                total_timesteps: 1_000_000,
                anneal_lr: true,
                gamma: 0.99,
                gae_lambda: 0.95,
                clip_eps: 0.2,
                vf_cost: 0.5,
                entropy_cost: 0.00,
                target_kl: None,
                normalize_advantages: true,
            },
            environment: EnvConfig {
                env_id: "Humanoid-v4".to_string(),
                num_envs: 8,
                normalize_observations: true,
                normalize_rewards: true,
                clip_observations: 10.0,
                clip_rewards: 10.0,
            },
            network: NetworkConfig {
                policy_hidden_sizes: vec![32, 32, 32, 32],
                value_hidden_sizes: vec![256, 256, 256, 256, 256],
                activation: "swish".to_string(),
                squash_distribution: true,
            },
            experiment_name: "ppo_experiment".to_string(),
            seed: 42,
            save_model: false,
            eval_every: 5,
            num_eval_episodes: 10,
            deterministic_eval: true,
        }
    }

    pub fn a2c_config() -> TrainingConfig {
        let mut config = Self::ppo_config();
        config.algorithm.update_epochs = 1; // A2C doesn't use multiple epochs
        config.algorithm.clip_eps = 0.0; // No clipping in A2C
        config.experiment_name = "a2c_experiment".to_string();
        config
    }

    pub fn reinforce_config() -> TrainingConfig {
        let mut config = Self::ppo_config();
        config.algorithm.normalize_advantages = false;
        config.algorithm.clip_eps = 0.0;
        config.algorithm.update_epochs = 1;
        config.experiment_name = "reinforce_experiment".to_string();
        config
    }
}
