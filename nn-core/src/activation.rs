//! Activation functions for artificial neural networks.
//!
//! This module exports several activation functions that can be used to activate neurons in a
//! neural network.

pub type ActivationFunction = fn(f32) -> f32;

/// Sigmoid activation function.
///
/// This function is often referred to as the logistic function. The function's domain is the entire
/// real range, and it returns a value between 0 and 1.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
