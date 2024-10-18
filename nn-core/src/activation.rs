//! Activation functions for artificial neural networks.
//!
//! This module exports several activation functions that can be used to activate neurons in a
//! neural network.

use std::fmt::Debug;

pub trait ActivationFunction: Debug {
    fn call(&self, x: f32) -> f32;
    fn derivative(&self, x: f32) -> f32;
}

/// Sigmoid activation function
#[derive(Debug)]
pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn call(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn derivative(&self, x: f32) -> f32 {
        x * (1.0 - x)
    }
}

/// Linear activation function
#[derive(Debug)]
pub struct Linear;

impl ActivationFunction for Linear {
    fn call(&self, x: f32) -> f32 {
        x
    }
    fn derivative(&self, _: f32) -> f32 {
        1.0
    }
}
