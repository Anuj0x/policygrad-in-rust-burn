//! Probability distributions for policy gradient methods

use burn::tensor::{backend::Backend, Tensor, Distribution};
use burn::prelude::*;
use rand::prelude::*;
use std::f32::consts::PI;

/// Base trait for parametric distributions
pub trait ParametricDistribution<B: Backend> {
    fn sample(&self, params: &Tensor<B, 2>, key: &mut impl Rng) -> Tensor<B, 2>;
    fn log_prob(&self, params: &Tensor<B, 2>, actions: &Tensor<B, 2>) -> Tensor<B, 2>;
    fn entropy(&self, params: &Tensor<B, 2>) -> Tensor<B, 1>;
    fn mode(&self, params: &Tensor<B, 2>) -> Tensor<B, 2>;
}

/// Normal distribution with tanh squashing for continuous action spaces
pub struct NormalTanhDistribution;

impl<B: Backend> ParametricDistribution<B> for NormalTanhDistribution {
    fn sample(&self, params: &Tensor<B, 2>, key: &mut impl Rng) -> Tensor<B, 2> {
        let [batch_size, param_size] = params.dims();
        let mean = params.clone().slice([0..batch_size, 0..param_size/2]);
        let log_std = params.clone().slice([0..batch_size, param_size/2..param_size]);

        // Sample from normal distribution
        let std = log_std.exp();
        let noise = Tensor::<B, 2>::random_normal([batch_size, param_size/2], Distribution::Normal(0.0, 1.0), key);
        let raw_actions = mean + std * noise;

        // Apply tanh squashing
        raw_actions.tanh()
    }

    fn log_prob(&self, params: &Tensor<B, 2>, actions: &Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, param_size] = params.dims();
        let mean = params.clone().slice([0..batch_size, 0..param_size/2]);
        let log_std = params.clone().slice([0..batch_size, param_size/2..param_size]);

        let std = log_std.exp();
        let var = std.powf(2.0);

        // Log probability of normal distribution
        let log_prob_normal = -0.5 * (
            (actions - mean).powf(2.0) / var +
            2.0 * log_std +
            (2.0 * PI).ln()
        );

        // Log determinant of Jacobian for tanh transformation
        let log_det_jacobian = (1.0 - actions.powf(2.0) + 1e-6).ln().sum_dim(1);

        log_prob_normal.sum_dim(1) - log_det_jacobian
    }

    fn entropy(&self, params: &Tensor<B, 2>) -> Tensor<B, 1> {
        let [batch_size, param_size] = params.dims();
        let log_std = params.clone().slice([0..batch_size, param_size/2..param_size]);

        // Entropy of normal distribution
        let entropy_normal = 0.5 + 0.5 * (2.0 * PI * log_std.exp().powf(2.0)).ln();

        // Account for tanh transformation (approximate)
        entropy_normal.sum_dim(1) + (param_size as f32 / 2.0) * (1.0 - 2.0/PI).ln()
    }

    fn mode(&self, params: &Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, param_size] = params.dims();
        let mean = params.clone().slice([0..batch_size, 0..param_size/2]);

        // Mode of tanh-transformed normal is tanh(mean)
        mean.tanh()
    }
}

/// Categorical distribution for discrete action spaces
pub struct CategoricalDistribution {
    num_actions: usize,
}

impl CategoricalDistribution {
    pub fn new(num_actions: usize) -> Self {
        Self { num_actions }
    }
}

impl<B: Backend> ParametricDistribution<B> for CategoricalDistribution {
    fn sample(&self, params: &Tensor<B, 2>, key: &mut impl Rng) -> Tensor<B, 2> {
        let logits = params.clone();
        let probs = logits.softmax(1);

        // Sample from categorical distribution
        let [batch_size, _] = probs.dims();
        let mut samples = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let row_probs = probs.clone().slice([i..i+1, 0..self.num_actions]);
            let row_probs_vec: Vec<f32> = row_probs.into_data().value;

            let mut cumsum = 0.0;
            let r: f32 = key.gen();
            let mut action = 0;

            for (j, &prob) in row_probs_vec.iter().enumerate() {
                cumsum += prob;
                if r <= cumsum {
                    action = j;
                    break;
                }
            }

            samples.push(action as f32);
        }

        Tensor::<B, 2>::from_data(samples.as_slice(), [batch_size, 1])
    }

    fn log_prob(&self, params: &Tensor<B, 2>, actions: &Tensor<B, 2>) -> Tensor<B, 2> {
        let logits = params.clone();
        let log_probs = logits.log_softmax(1);

        // Gather log probabilities for selected actions
        let actions_int: Vec<usize> = actions.clone()
            .into_data()
            .value
            .iter()
            .map(|&x| x as usize)
            .collect();

        let mut result = Vec::with_capacity(actions_int.len());

        for (i, &action) in actions_int.iter().enumerate() {
            let log_prob = log_probs.clone().slice([i..i+1, action..action+1]);
            result.push(log_prob.into_scalar());
        }

        Tensor::<B, 2>::from_data(result.as_slice(), actions.dims())
    }

    fn entropy(&self, params: &Tensor<B, 2>) -> Tensor<B, 1> {
        let logits = params.clone();
        let probs = logits.softmax(1);
        let log_probs = probs.clone().log();

        -(probs * log_probs).sum_dim(1)
    }

    fn mode(&self, params: &Tensor<B, 2>) -> Tensor<B, 2> {
        // Mode is the action with highest probability
        params.clone().argmax(1).float().unsqueeze_dim(1)
    }
}

/// Standard normal distribution for continuous spaces
pub struct NormalDistribution;

impl<B: Backend> ParametricDistribution<B> for NormalDistribution {
    fn sample(&self, params: &Tensor<B, 2>, key: &mut impl Rng) -> Tensor<B, 2> {
        let [batch_size, param_size] = params.dims();
        let mean = params.clone().slice([0..batch_size, 0..param_size/2]);
        let log_std = params.clone().slice([0..batch_size, param_size/2..param_size]);

        let std = log_std.exp();
        let noise = Tensor::<B, 2>::random_normal([batch_size, param_size/2], Distribution::Normal(0.0, 1.0), key);

        mean + std * noise
    }

    fn log_prob(&self, params: &Tensor<B, 2>, actions: &Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, param_size] = params.dims();
        let mean = params.clone().slice([0..batch_size, 0..param_size/2]);
        let log_std = params.clone().slice([0..batch_size, param_size/2..param_size]);

        let std = log_std.exp();
        let var = std.powf(2.0);

        -0.5 * (
            (actions - mean).powf(2.0) / var +
            2.0 * log_std +
            (2.0 * PI).ln()
        ).sum_dim(1)
    }

    fn entropy(&self, params: &Tensor<B, 2>) -> Tensor<B, 1> {
        let [batch_size, param_size] = params.dims();
        let log_std = params.clone().slice([0..batch_size, param_size/2..param_size]);

        // Entropy = 0.5 * dim * (1 + ln(2π)) + sum(ln(σ))
        let dim = param_size as f32 / 2.0;
        0.5 * dim * (1.0 + (2.0 * PI).ln()) + log_std.sum_dim(1)
    }

    fn mode(&self, params: &Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, param_size] = params.dims();
        params.clone().slice([0..batch_size, 0..param_size/2])
    }
}
