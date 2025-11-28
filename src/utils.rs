//! Utility functions and helpers

use burn::tensor::{backend::Backend, Tensor};
use rand::prelude::*;
use std::time::{Duration, Instant};

/// Timing utilities for performance measurement
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Random number generation utilities
pub struct RandomUtils;

impl RandomUtils {
    /// Set global random seed
    pub fn set_seed(seed: u64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        rand::rngs::ThreadRng::default().fill(&mut rng);
    }

    /// Generate random tensor with normal distribution
    pub fn randn<B: Backend>(shape: &[usize], mean: f32, std: f32) -> Tensor<B, 2> {
        let total_elements = shape.iter().product();
        let mut data = vec![0.0f32; total_elements];

        let mut rng = rand::thread_rng();
        for x in data.iter_mut() {
            *x = rng.gen::<f32>() * std + mean;
        }

        Tensor::from_data(data.as_slice(), shape)
    }

    /// Generate random tensor with uniform distribution
    pub fn rand<B: Backend>(shape: &[usize], low: f32, high: f32) -> Tensor<B, 2> {
        let total_elements = shape.iter().product();
        let mut data = vec![0.0f32; total_elements];

        let mut rng = rand::thread_rng();
        for x in data.iter_mut() {
            *x = rng.gen::<f32>() * (high - low) + low;
        }

        Tensor::from_data(data.as_slice(), shape)
    }
}

/// Mathematical utilities
pub struct MathUtils;

impl MathUtils {
    /// Softmax function
    pub fn softmax<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let max_vals = x.clone().max_dim(1);
        let exp_x = (x - max_vals).exp();
        let sum_exp = exp_x.clone().sum_dim(1);
        exp_x / sum_exp
    }

    /// Log softmax function (numerically stable)
    pub fn log_softmax<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let max_vals = x.clone().max_dim(1);
        let exp_x = (x - max_vals).exp();
        let log_sum_exp = exp_x.sum_dim(1).log();
        x - max_vals - log_sum_exp
    }

    /// Sigmoid function
    pub fn sigmoid<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
        1.0 / (1.0 + (-x).exp())
    }

    /// Tanh function
    pub fn tanh<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
        x.tanh()
    }

    /// ReLU function
    pub fn relu<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
        x.relu()
    }

    /// Swish activation: x * sigmoid(x)
    pub fn swish<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
        x.clone() * Self::sigmoid(x)
    }

    /// Compute mean and standard deviation
    pub fn mean_std<B: Backend>(x: &Tensor<B, 2>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let mean = x.mean_dim(0);
        let var = x.var_dim(0, false);
        let std = var.sqrt();
        (mean, std)
    }

    /// Normalize tensor to zero mean and unit variance
    pub fn normalize<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let (mean, std) = Self::mean_std(x);
        (x - mean) / (std + 1e-8)
    }

    /// Clip tensor values to range
    pub fn clip<B: Backend>(x: &Tensor<B, 2>, min_val: f32, max_val: f32) -> Tensor<B, 2> {
        x.clamp(min_val, max_val)
    }
}

/// Logging utilities
pub struct LoggingUtils;

impl LoggingUtils {
    /// Format duration as human-readable string
    pub fn format_duration(duration: Duration) -> String {
        let total_seconds = duration.as_secs_f64();

        if total_seconds < 60.0 {
            format!("{:.2}s", total_seconds)
        } else if total_seconds < 3600.0 {
            let minutes = (total_seconds / 60.0).floor();
            let seconds = total_seconds % 60.0;
            format!("{:.0}m {:.1}s", minutes, seconds)
        } else {
            let hours = (total_seconds / 3600.0).floor();
            let minutes = ((total_seconds % 3600.0) / 60.0).floor();
            format!("{:.0}h {:.0}m", hours, minutes)
        }
    }

    /// Format large numbers with appropriate units
    pub fn format_large_number(num: f64) -> String {
        if num >= 1e9 {
            format!("{:.2}B", num / 1e9)
        } else if num >= 1e6 {
            format!("{:.2}M", num / 1e6)
        } else if num >= 1e3 {
            format!("{:.2}K", num / 1e3)
        } else {
            format!("{:.2}", num)
        }
    }

    /// Create progress bar string
    pub fn progress_bar(current: usize, total: usize, width: usize) -> String {
        let progress = current as f32 / total as f32;
        let filled = (progress * width as f32) as usize;
        let empty = width - filled;

        let filled_bar = "█".repeat(filled);
        let empty_bar = "░".repeat(empty);
        let percent = (progress * 100.0) as usize;

        format!("[{}{}] {}%", filled_bar, empty_bar, percent)
    }
}

/// File I/O utilities
pub struct FileUtils;

impl FileUtils {
    /// Create directory if it doesn't exist
    pub fn ensure_dir(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        Ok(())
    }

    /// Save tensor to binary file
    pub fn save_tensor<B: Backend>(
        tensor: &Tensor<B, 2>,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Self::ensure_dir(path.parent().unwrap())?;
        let data = tensor.clone().into_data().value;
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();

        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load tensor from binary file
    pub fn load_tensor<B: Backend>(
        shape: &[usize],
        path: &std::path::Path,
    ) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let total_elements = shape.iter().product();

        if bytes.len() != total_elements * std::mem::size_of::<f32>() {
            return Err("File size doesn't match tensor shape".into());
        }

        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Tensor::from_data(data.as_slice(), shape))
    }
}

/// Performance benchmarking utilities
pub struct BenchmarkUtils;

impl BenchmarkUtils {
    /// Benchmark a function and return execution time
    pub fn benchmark<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Benchmark a function multiple times and return statistics
    pub fn benchmark_stats<F, R>(f: F, num_runs: usize) -> BenchmarkStats
    where
        F: Fn() -> R,
    {
        let mut times = Vec::with_capacity(num_runs);

        for _ in 0..num_runs {
            let (_, duration) = Self::benchmark(&f);
            times.push(duration.as_secs_f64());
        }

        let mean = times.iter().sum::<f64>() / num_runs as f64;
        let variance = times.iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f64>() / num_runs as f64;
        let std = variance.sqrt();

        let min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        BenchmarkStats {
            mean,
            std,
            min,
            max,
            num_runs,
        }
    }
}

/// Benchmark statistics
pub struct BenchmarkStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub num_runs: usize,
}

impl BenchmarkStats {
    pub fn report(&self, name: &str) {
        println!("Benchmark: {}", name);
        println!("  Mean: {:.4}s (±{:.4}s)", self.mean, self.std);
        println!("  Min:  {:.4}s", self.min);
        println!("  Max:  {:.4}s", self.max);
        println!("  Runs: {}", self.num_runs);
    }
}
