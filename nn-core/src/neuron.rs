//! Representation of a neuron in a neural network.
//!
//! This module's implementation of a neuron is rather simplistic. It has a number of input
//! parameters, each having a weight, a single bias, and an activation function associated with it.
//!
//! The neuron also provides an output value, a delta value for networks that use backpropagation,
//! and a method to update the weights based on the delta value.

use crate::activation::ActivationFunction;
use crate::tensor::{Tensor, Tensor1D};

/// A single neuron in a neural network.
///
/// The neuron holds its input edges in the network in the form of inputs, it maintains a separate
/// weight for each input edge, a shared bias, and an activation function.
#[derive(Debug)]
pub struct Neuron {
    /// 1-dimensional tensor of input values
    input: Tensor1D,
    /// 1-dimensional tensor of neuron weights, one for each input edge
    weights: Tensor1D,
    bias: f32,
    activation: &'static dyn ActivationFunction,
    output: f32,
    delta: f32,
}

impl Neuron {
    /// Create a new neuron with the given input size and activation function.
    ///
    /// By default, the bias is set to zero, and a uniform distribution across [-0.5, 0.5] is used
    /// for the weights.
    ///
    /// These standard options can be changed by calling the `set_bias`, `set_weights` methods.
    pub fn new(input_size: usize, activation: &'static impl ActivationFunction) -> Self {
        Self {
            input: Tensor::zeros([input_size]),
            weights: Tensor::uniform([input_size], -0.5, 0.5),
            bias: rand::random::<f32>() % 1.0,
            output: 0.0,
            delta: 0.0,
            activation,
        }
    }

    /// Make a forward pass through the neuron.
    pub fn forward(&mut self, input: &Tensor1D) -> f32 {
        let linear = input.inner_product(&self.weights) + self.bias;
        self.output = self.activation.call(linear);
        self.output
    }

    /// Update the weights and bias of the neuron according to the given learning rate.
    pub fn update_weights(&mut self, learning_rate: f32) {
        self.weights.scale(learning_rate);
        self.weights.scale(self.delta);
        self.bias *= learning_rate * self.delta;
    }

    /// Get the weights of the neuron
    pub fn weights(&self) -> &Tensor1D {
        &self.weights
    }
}
