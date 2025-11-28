//! Proximal Policy Optimization (PPO) implementation

use burn::module::{Module, Param};
use burn::nn::loss::Reduction;
use burn::optim::Optimizer;
use burn::tensor::{backend::Backend, Tensor};
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};

use crate::distributions::{NormalTanhDistribution, CategoricalDistribution, ParametricDistribution};
use crate::networks::{ContinuousPolicyNetwork, DiscretePolicyNetwork, ValueNetwork, Activation};
use crate::training::{AlgorithmConfig, TrainingMetrics};

/// PPO network parameters
#[derive(Module, Debug)]
pub struct PPONetworkParams<B: Backend> {
    pub policy: Param<Tensor<B, 2>>,
    pub value: Param<Tensor<B, 2>>,
}

/// PPO networks combining policy and value networks
pub struct PPONetworks<B: Backend, D: ParametricDistribution<B>> {
    pub policy_network: ContinuousPolicyNetwork<B>,
    pub value_network: ValueNetwork<B>,
    pub action_distribution: D,
}

impl<B: Backend, D: ParametricDistribution<B>> PPONetworks<B, D> {
    pub fn new(
        observation_size: usize,
        action_size: usize,
        policy_hidden_sizes: &[usize],
        value_hidden_sizes: &[usize],
        activation: Activation,
        action_distribution: D,
    ) -> Self {
        let policy_network = ContinuousPolicyNetwork::new(
            observation_size,
            action_size,
            policy_hidden_sizes,
            activation,
            true, // squash distribution
        );

        let value_network = ValueNetwork::new(
            observation_size,
            value_hidden_sizes,
            activation,
        );

        Self {
            policy_network,
            value_network,
            action_distribution,
        }
    }

    pub fn forward_policy(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        self.policy_network.forward(observation)
    }

    pub fn forward_value(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        self.value_network.forward(observation)
    }
}

/// PPO training state
pub struct PPOTrainingState<B: Backend> {
    pub networks: PPONetworks<B, NormalTanhDistribution>,
    pub optimizer_state: <burn::optim::Adam<B> as Optimizer<B>>::State,
}

/// PPO rollout data structure
#[derive(Clone, Debug)]
pub struct RolloutData<B: Backend> {
    pub observations: Tensor<B, 3>,      // [batch_size, unroll_length, obs_dim]
    pub actions: Tensor<B, 3>,           // [batch_size, unroll_length, action_dim]
    pub rewards: Tensor<B, 3>,           // [batch_size, unroll_length, 1]
    pub dones: Tensor<B, 3>,             // [batch_size, unroll_length, 1]
    pub log_probs: Tensor<B, 3>,         // [batch_size, unroll_length, 1]
    pub values: Tensor<B, 3>,            // [batch_size, unroll_length, 1]
    pub advantages: Tensor<B, 3>,        // [batch_size, unroll_length, 1]
    pub returns: Tensor<B, 3>,           // [batch_size, unroll_length, 1]
}

/// PPO loss computation
pub fn compute_ppo_loss<B: Backend>(
    networks: &PPONetworks<B, NormalTanhDistribution>,
    data: &RolloutData<B>,
    config: &AlgorithmConfig,
) -> (Tensor<B, 1>, TrainingMetrics) {
    // Flatten batch and time dimensions
    let batch_size = data.observations.dims()[0];
    let unroll_length = data.observations.dims()[1];
    let total_samples = batch_size * unroll_length;

    let observations = data.observations.clone().reshape([total_samples, -1]);
    let actions = data.actions.clone().reshape([total_samples, -1]);
    let old_log_probs = data.log_probs.clone().reshape([total_samples, -1]);
    let advantages = data.advantages.clone().reshape([total_samples, -1]);
    let returns = data.returns.clone().reshape([total_samples, -1]);
    let values = data.values.clone().reshape([total_samples, -1]);

    // Forward pass
    let policy_params = networks.forward_policy(observations.clone());
    let current_values = networks.forward_value(observations.clone());

    // Policy loss (PPO clipped objective)
    let new_log_probs = networks.action_distribution.log_prob(&policy_params, &actions);
    let ratio = (new_log_probs - old_log_probs).exp();

    let surr1 = ratio * advantages.clone();
    let surr2 = ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages.clone();

    let policy_loss = -surr1.min(surr2).mean();

    // Value loss (MSE)
    let value_loss = (current_values - returns).powf(2.0).mean() * config.vf_cost;

    // Entropy bonus
    let entropy = networks.action_distribution.entropy(&policy_params).mean();
    let entropy_loss = -entropy * config.entropy_cost;

    // Total loss
    let total_loss = policy_loss + value_loss + entropy_loss;

    // Compute additional metrics
    let approx_kl = ((ratio - 1.0) - (new_log_probs - old_log_probs)).mean();
    let clip_fraction = (ratio.abs() > config.clip_eps as f32).float().mean();

    let metrics = TrainingMetrics {
        policy_loss: policy_loss.into_scalar(),
        value_loss: value_loss.into_scalar(),
        entropy_loss: entropy_loss.into_scalar(),
        entropy: entropy.into_scalar(),
        approx_kl: approx_kl.into_scalar(),
        clip_fraction: clip_fraction.into_scalar(),
        ..Default::default()
    };

    (total_loss, metrics)
}

/// PPO training step
impl<B: Backend> TrainStep<RolloutData<B>, TrainingMetrics> for PPOTrainingState<B> {
    fn step(&mut self, data: RolloutData<B>) -> TrainOutput<TrainingMetrics> {
        let config = AlgorithmConfig {
            clip_eps: 0.2,
            vf_cost: 0.5,
            entropy_cost: 0.00,
            normalize_advantages: true,
            ..Default::default()
        };

        let (loss, metrics) = compute_ppo_loss(&self.networks, &data, &config);

        // Backward pass and optimization would be handled by the trainer
        TrainOutput::new(self, loss, metrics)
    }
}

/// PPO validation step
impl<B: Backend> ValidStep<RolloutData<B>, TrainingMetrics> for PPOTrainingState<B> {
    fn step(&self, data: RolloutData<B>) -> TrainingMetrics {
        let config = AlgorithmConfig {
            clip_eps: 0.2,
            vf_cost: 0.5,
            entropy_cost: 0.00,
            normalize_advantages: true,
            ..Default::default()
        };

        let (_, metrics) = compute_ppo_loss(&self.networks, &data, &config);
        metrics
    }
}

/// PPO algorithm implementation
pub struct PPO<B: Backend> {
    networks: PPONetworks<B, NormalTanhDistribution>,
}

impl<B: Backend> PPO<B> {
    pub fn new(
        observation_size: usize,
        action_size: usize,
        policy_hidden_sizes: &[usize],
        value_hidden_sizes: &[usize],
        activation: Activation,
    ) -> Self {
        let networks = PPONetworks::new(
            observation_size,
            action_size,
            policy_hidden_sizes,
            value_hidden_sizes,
            activation,
            NormalTanhDistribution,
        );

        Self { networks }
    }

    pub fn act(&self, observation: &Tensor<B, 2>, deterministic: bool) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let policy_params = self.networks.forward_policy(observation.clone());
        let value = self.networks.forward_value(observation.clone());

        let action = if deterministic {
            self.networks.action_distribution.mode(&policy_params)
        } else {
            self.networks.action_distribution.sample(&policy_params, &mut rand::thread_rng())
        };

        let log_prob = self.networks.action_distribution.log_prob(&policy_params, &action);

        (action, log_prob)
    }

    pub fn value(&self, observation: &Tensor<B, 2>) -> Tensor<B, 2> {
        self.networks.forward_value(observation.clone())
    }

    pub fn networks(&self) -> &PPONetworks<B, NormalTanhDistribution> {
        &self.networks
    }

    pub fn networks_mut(&mut self) -> &mut PPONetworks<B, NormalTanhDistribution> {
        &mut self.networks
    }
}

/// Discrete PPO variant for discrete action spaces
pub struct DiscretePPO<B: Backend> {
    networks: PPONetworks<B, CategoricalDistribution>,
}

impl<B: Backend> DiscretePPO<B> {
    pub fn new(
        observation_size: usize,
        num_actions: usize,
        policy_hidden_sizes: &[usize],
        value_hidden_sizes: &[usize],
        activation: Activation,
    ) -> Self {
        let networks = PPONetworks::new(
            observation_size,
            num_actions,
            policy_hidden_sizes,
            value_hidden_sizes,
            activation,
            CategoricalDistribution::new(num_actions),
        );

        Self { networks }
    }

    pub fn act(&self, observation: &Tensor<B, 2>, deterministic: bool) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let policy_params = self.networks.forward_policy(observation.clone());
        let value = self.networks.forward_value(observation.clone());

        let action = if deterministic {
            self.networks.action_distribution.mode(&policy_params)
        } else {
            self.networks.action_distribution.sample(&policy_params, &mut rand::thread_rng())
        };

        let log_prob = self.networks.action_distribution.log_prob(&policy_params, &action);

        (action, log_prob)
    }

    pub fn value(&self, observation: &Tensor<B, 2>) -> Tensor<B, 2> {
        self.networks.forward_value(observation.clone())
    }

    pub fn networks(&self) -> &PPONetworks<B, CategoricalDistribution> {
        &self.networks
    }

    pub fn networks_mut(&mut self) -> &mut PPONetworks<B, CategoricalDistribution> {
        &mut self.networks
    }
}
