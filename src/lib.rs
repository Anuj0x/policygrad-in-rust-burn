//! # Policy Gradients in Rust
//!
//! A high-performance, memory-safe implementation of on-policy policy gradient algorithms
//! using the Burn deep learning framework.
//!
//! ## Features
//!
//! - **Zero-cost abstractions**: Rust's ownership system ensures memory safety without runtime overhead
//! - **SIMD acceleration**: Burn leverages platform-specific optimizations
//! - **Modular design**: Clean separation between algorithms, networks, and environments
//! - **GPU acceleration**: WGPU backend for high-performance computing
//! - **Type safety**: Compile-time guarantees prevent runtime errors

pub mod algorithms;
pub mod environments;
pub mod networks;
pub mod distributions;
pub mod training;
pub mod utils;

pub use algorithms::*;
pub use environments::*;
pub use networks::*;
pub use distributions::*;
pub use training::*;
