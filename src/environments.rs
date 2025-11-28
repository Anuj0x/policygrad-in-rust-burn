//! Environment interfaces for reinforcement learning

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, IxDyn};
use std::collections::VecDeque;

/// Core observation type - supports various shapes and data types
pub type Observation = ArrayD<f32>;

/// Core action type
pub type Action = ArrayD<f32>;

/// Environment info structure
#[derive(Clone, Debug)]
pub struct EnvInfo {
    pub episode_return: f32,
    pub episode_length: usize,
    pub truncated: bool,
    pub terminated: bool,
}

/// Transition structure for experience replay
#[derive(Clone, Debug)]
pub struct Transition {
    pub observation: Observation,
    pub action: Action,
    pub reward: f32,
    pub next_observation: Observation,
    pub done: bool,
    pub info: EnvInfo,
}

/// Base environment trait
pub trait Environment: Send + Sync {
    /// Reset environment and return initial observation
    fn reset(&mut self) -> Result<Observation, Box<dyn std::error::Error>>;

    /// Take an action and return (observation, reward, done, info)
    fn step(&mut self, action: &Action) -> Result<(Observation, f32, bool, EnvInfo), Box<dyn std::error::Error>>;

    /// Get observation space shape
    fn observation_space(&self) -> Vec<usize>;

    /// Get action space shape
    fn action_space(&self) -> Vec<usize>;

    /// Check if action space is discrete
    fn is_discrete_action(&self) -> bool;

    /// Get number of actions (for discrete spaces)
    fn num_actions(&self) -> Option<usize>;

    /// Seed the environment
    fn seed(&mut self, seed: u64);

    /// Close the environment
    fn close(&mut self);
}

/// Batched environment for parallel rollouts
pub trait BatchedEnvironment: Send + Sync {
    /// Reset all environments and return initial observations
    fn reset(&mut self) -> Result<Vec<Observation>, Box<dyn std::error::Error>>;

    /// Take actions in all environments
    fn step(&mut self, actions: &[Action]) -> Result<Vec<(Observation, f32, bool, EnvInfo)>, Box<dyn std::error::Error>>;

    /// Get number of environments in batch
    fn num_envs(&self) -> usize;

    /// Seed all environments
    fn seed(&mut self, seed: u64);

    /// Close all environments
    fn close(&mut self);
}

/// Vectorized environment implementation
pub struct VecEnv<E: Environment> {
    envs: Vec<E>,
    observations: Vec<Observation>,
}

impl<E: Environment> VecEnv<E> {
    pub fn new(envs: Vec<E>) -> Self {
        Self {
            envs,
            observations: Vec::new(),
        }
    }
}

impl<E: Environment> BatchedEnvironment for VecEnv<E> {
    fn reset(&mut self) -> Result<Vec<Observation>, Box<dyn std::error::Error>> {
        let mut observations = Vec::with_capacity(self.envs.len());
        for env in &mut self.envs {
            observations.push(env.reset()?);
        }
        self.observations = observations.clone();
        Ok(observations)
    }

    fn step(&mut self, actions: &[Action]) -> Result<Vec<(Observation, f32, bool, EnvInfo)>, Box<dyn std::error::Error>> {
        assert_eq!(actions.len(), self.envs.len());

        let mut results = Vec::with_capacity(self.envs.len());
        for (env, action) in self.envs.iter_mut().zip(actions) {
            let result = env.step(action)?;
            results.push(result);
        }

        // Update cached observations
        for (i, (obs, _, _, _)) in results.iter().enumerate() {
            self.observations[i] = obs.clone();
        }

        Ok(results)
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn seed(&mut self, seed: u64) {
        for (i, env) in self.envs.iter_mut().enumerate() {
            env.seed(seed + i as u64);
        }
    }

    fn close(&mut self) {
        for env in &mut self.envs {
            env.close();
        }
    }
}

/// Environment wrapper for normalization
pub struct NormalizedEnv<E: Environment> {
    env: E,
    obs_mean: Option<ndarray::Array1<f32>>,
    obs_std: Option<ndarray::Array1<f32>>,
    obs_running_mean: ndarray::Array1<f32>,
    obs_running_std: ndarray::Array1<f32>,
    obs_count: usize,
    normalize_obs: bool,
    normalize_reward: bool,
    reward_running_mean: f32,
    reward_running_std: f32,
    reward_count: usize,
    clip_obs: f32,
    clip_reward: f32,
}

impl<E: Environment> NormalizedEnv<E> {
    pub fn new(
        env: E,
        normalize_obs: bool,
        normalize_reward: bool,
        clip_obs: f32,
        clip_reward: f32,
    ) -> Self {
        let obs_shape = env.observation_space();
        let obs_size = obs_shape.iter().product();

        Self {
            env,
            obs_mean: None,
            obs_std: None,
            obs_running_mean: ndarray::Array1::zeros(obs_size),
            obs_running_std: ndarray::Array1::ones(obs_size),
            obs_count: 0,
            normalize_obs,
            normalize_reward,
            reward_running_mean: 0.0,
            reward_running_std: 1.0,
            reward_count: 0,
            clip_obs,
            clip_reward,
        }
    }

    fn normalize_observation(&self, obs: &Observation) -> Observation {
        if !self.normalize_obs {
            return obs.clone();
        }

        let mut normalized = obs.clone();
        let flat_obs = normalized.as_slice_mut().unwrap();

        for i in 0..flat_obs.len() {
            flat_obs[i] = (flat_obs[i] - self.obs_running_mean[i]) / (self.obs_running_std[i] + 1e-8);
            flat_obs[i] = flat_obs[i].clamp(-self.clip_obs, self.clip_obs);
        }

        normalized
    }

    fn update_obs_stats(&mut self, obs: &Observation) {
        if !self.normalize_obs {
            return;
        }

        let flat_obs = obs.as_slice().unwrap();
        self.obs_count += 1;

        let alpha = 1.0 / self.obs_count as f32;

        for i in 0..flat_obs.len() {
            let diff = flat_obs[i] - self.obs_running_mean[i];
            self.obs_running_mean[i] += alpha * diff;
            self.obs_running_std[i] = (1.0 - alpha) * self.obs_running_std[i] + alpha * diff.abs();
        }
    }

    fn normalize_reward(&mut self, reward: f32) -> f32 {
        if !self.normalize_reward {
            return reward.clamp(-self.clip_reward, self.clip_reward);
        }

        self.reward_count += 1;
        let alpha = 1.0 / self.reward_count as f32;

        let diff = reward - self.reward_running_mean;
        self.reward_running_mean += alpha * diff;
        self.reward_running_std = (1.0 - alpha) * self.reward_running_std + alpha * diff.abs();

        let normalized_reward = (reward - self.reward_running_mean) / (self.reward_running_std + 1e-8);
        normalized_reward.clamp(-self.clip_reward, self.clip_reward)
    }
}

impl<E: Environment> Environment for NormalizedEnv<E> {
    fn reset(&mut self) -> Result<Observation, Box<dyn std::error::Error>> {
        let obs = self.env.reset()?;
        self.update_obs_stats(&obs);
        Ok(self.normalize_observation(&obs))
    }

    fn step(&mut self, action: &Action) -> Result<(Observation, f32, bool, EnvInfo), Box<dyn std::error::Error>> {
        let (obs, reward, done, info) = self.env.step(action)?;
        self.update_obs_stats(&obs);
        let normalized_obs = self.normalize_observation(&obs);
        let normalized_reward = self.normalize_reward(reward);
        Ok((normalized_obs, normalized_reward, done, info))
    }

    fn observation_space(&self) -> Vec<usize> {
        self.env.observation_space()
    }

    fn action_space(&self) -> Vec<usize> {
        self.env.action_space()
    }

    fn is_discrete_action(&self) -> bool {
        self.env.is_discrete_action()
    }

    fn num_actions(&self) -> Option<usize> {
        self.env.num_actions()
    }

    fn seed(&mut self, seed: u64) {
        self.env.seed(seed);
    }

    fn close(&mut self) {
        self.env.close();
    }
}

/// Environment factory for creating different environment types
pub struct EnvironmentFactory;

impl EnvironmentFactory {
    /// Create a normalized batched environment
    pub fn create_normalized_vec_env<E: Environment + Clone + 'static>(
        env_template: E,
        num_envs: usize,
        normalize_obs: bool,
        normalize_reward: bool,
        clip_obs: f32,
        clip_reward: f32,
    ) -> VecEnv<NormalizedEnv<E>> {
        let envs = (0..num_envs)
            .map(|_| NormalizedEnv::new(
                env_template.clone(),
                normalize_obs,
                normalize_reward,
                clip_obs,
                clip_reward,
            ))
            .collect();

        VecEnv::new(envs)
    }

    /// Create a basic batched environment
    pub fn create_vec_env<E: Environment + 'static>(
        envs: Vec<E>,
    ) -> VecEnv<E> {
        VecEnv::new(envs)
    }
}

/// Experience buffer for on-policy algorithms
pub struct ExperienceBuffer {
    buffer: VecDeque<Transition>,
    max_size: usize,
}

impl ExperienceBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
    }

    pub fn extend(&mut self, transitions: impl IntoIterator<Item = Transition>) {
        for transition in transitions {
            self.push(transition);
        }
    }

    pub fn sample_batch(&self, batch_size: usize) -> Vec<&Transition> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        self.buffer
            .iter()
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Transition> {
        self.buffer.iter()
    }
}
