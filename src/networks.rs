//! Neural network architectures using Burn

use burn::nn::{Linear, LinearConfig, Module, ReLU, Tanh, Swish};
use burn::tensor::{backend::Backend, Tensor};
use burn::prelude::*;

/// Activation function types
#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    Tanh,
    Swish,
}

impl Activation {
    pub fn forward<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Activation::ReLU => ReLU.forward(x),
            Activation::Tanh => Tanh.forward(x),
            Activation::Swish => x.clone() * x.sigmoid(), // Swish: x * sigmoid(x)
        }
    }
}

/// Feed-forward neural network module
#[derive(Module, Debug)]
pub struct FeedForwardNetwork<B: Backend> {
    layers: Vec<Linear<B>>,
    activations: Vec<Activation>,
    output_activation: Option<Activation>,
}

impl<B: Backend> FeedForwardNetwork<B> {
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        activation: Activation,
        output_activation: Option<Activation>,
    ) -> Self {
        let mut layers = Vec::new();
        let mut activations = Vec::new();

        // Input layer
        let mut prev_size = input_size;
        for &hidden_size in hidden_sizes {
            layers.push(
                LinearConfig::new(prev_size, hidden_size)
                    .init(std::default::Default::default())
            );
            activations.push(activation);
            prev_size = hidden_size;
        }

        // Output layer
        layers.push(
            LinearConfig::new(prev_size, output_size)
                .init(std::default::Default::default())
        );

        Self {
            layers,
            activations,
            output_activation,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        // Hidden layers
        for (layer, &activation) in self.layers.iter().zip(&self.activations) {
            x = layer.forward(x);
            x = activation.forward(x);
        }

        // Output layer
        x = self.layers.last().unwrap().forward(x);
        if let Some(act) = self.output_activation {
            x = act.forward(x);
        }

        x
    }
}

/// Policy network for discrete action spaces
#[derive(Module, Debug)]
pub struct DiscretePolicyNetwork<B: Backend> {
    network: FeedForwardNetwork<B>,
}

impl<B: Backend> DiscretePolicyNetwork<B> {
    pub fn new(
        observation_size: usize,
        num_actions: usize,
        hidden_sizes: &[usize],
        activation: Activation,
    ) -> Self {
        Self {
            network: FeedForwardNetwork::new(
                observation_size,
                hidden_sizes,
                num_actions,
                activation,
                None, // Raw logits for categorical distribution
            ),
        }
    }

    pub fn forward(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        self.network.forward(observation)
    }
}

/// Policy network for continuous action spaces
#[derive(Module, Debug)]
pub struct ContinuousPolicyNetwork<B: Backend> {
    network: FeedForwardNetwork<B>,
    action_size: usize,
}

impl<B: Backend> ContinuousPolicyNetwork<B> {
    pub fn new(
        observation_size: usize,
        action_size: usize,
        hidden_sizes: &[usize],
        activation: Activation,
        squash_distribution: bool,
    ) -> Self {
        let output_activation = if squash_distribution {
            None // Let distribution handle squashing
        } else {
            None
        };

        Self {
            network: FeedForwardNetwork::new(
                observation_size,
                hidden_sizes,
                action_size * 2, // mean and log_std
                activation,
                output_activation,
            ),
            action_size,
        }
    }

    pub fn forward(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        self.network.forward(observation)
    }
}

/// Value network for state value estimation
#[derive(Module, Debug)]
pub struct ValueNetwork<B: Backend> {
    network: FeedForwardNetwork<B>,
}

impl<B: Backend> ValueNetwork<B> {
    pub fn new(
        observation_size: usize,
        hidden_sizes: &[usize],
        activation: Activation,
    ) -> Self {
        Self {
            network: FeedForwardNetwork::new(
                observation_size,
                hidden_sizes,
                1, // Scalar value
                activation,
                None,
            ),
        }
    }

    pub fn forward(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        self.network.forward(observation)
    }
}

/// Atari feature extractor (CNN for image observations)
#[derive(Module, Debug)]
pub struct AtariFeatureExtractor<B: Backend> {
    conv1: burn::nn::Conv2d<B>,
    conv2: burn::nn::Conv2d<B>,
    conv3: burn::nn::Conv2d<B>,
    pool: burn::nn::MaxPool2d,
    flatten: burn::nn::Flatten,
    fc: Linear<B>,
}

impl<B: Backend> AtariFeatureExtractor<B> {
    pub fn new(
        input_channels: usize,
        dense_hidden_size: usize,
        activation: Activation,
    ) -> Self {
        Self {
            conv1: burn::nn::Conv2dConfig::new([input_channels, 32], [8, 8])
                .with_stride([4, 4])
                .init(std::default::Default::default()),
            conv2: burn::nn::Conv2dConfig::new([32, 64], [4, 4])
                .with_stride([2, 2])
                .init(std::default::Default::default()),
            conv3: burn::nn::Conv2dConfig::new([64, 64], [3, 3])
                .with_stride([1, 1])
                .init(std::default::Default::default()),
            pool: burn::nn::MaxPool2dConfig::new([2, 2]).init(),
            flatten: burn::nn::FlattenConfig::new().init(),
            fc: LinearConfig::new(3136, dense_hidden_size) // Adjust based on input size
                .init(std::default::Default::default()),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input);
        let x = Activation::ReLU.forward(x);

        let x = self.conv2.forward(x);
        let x = Activation::ReLU.forward(x);

        let x = self.conv3.forward(x);
        let x = Activation::ReLU.forward(x);

        let x = self.flatten.forward(x);
        let x = self.fc.forward(x);
        Activation::ReLU.forward(x)
    }
}

/// Network factory functions
pub struct NetworkFactory;

impl NetworkFactory {
    pub fn create_discrete_policy_network<B: Backend>(
        observation_size: usize,
        num_actions: usize,
        hidden_sizes: &[usize],
        activation: Activation,
    ) -> DiscretePolicyNetwork<B> {
        DiscretePolicyNetwork::new(observation_size, num_actions, hidden_sizes, activation)
    }

    pub fn create_continuous_policy_network<B: Backend>(
        observation_size: usize,
        action_size: usize,
        hidden_sizes: &[usize],
        activation: Activation,
        squash_distribution: bool,
    ) -> ContinuousPolicyNetwork<B> {
        ContinuousPolicyNetwork::new(observation_size, action_size, hidden_sizes, activation, squash_distribution)
    }

    pub fn create_value_network<B: Backend>(
        observation_size: usize,
        hidden_sizes: &[usize],
        activation: Activation,
    ) -> ValueNetwork<B> {
        ValueNetwork::new(observation_size, hidden_sizes, activation)
    }

    pub fn create_atari_feature_extractor<B: Backend>(
        input_channels: usize,
        dense_hidden_size: usize,
        activation: Activation,
    ) -> AtariFeatureExtractor<B> {
        AtariFeatureExtractor::new(input_channels, dense_hidden_size, activation)
    }
}
