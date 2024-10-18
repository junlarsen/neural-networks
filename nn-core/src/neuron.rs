//! Representation of a neuron in a neural network.
//!
//! This module's implementation of a neuron is rather simplistic. It has a number of input
//! parameters, each having a weight, a single bias, and an activation function associated with it.
//!
//! The neuron also provides an output value, a delta value for networks that use backpropagation,
//! and a method to update the weights based on the delta value.

use crate::activation::ActivationFunction;
use crate::tensor::{vector, Tensor};

/// A single neuron in a neural network.
///
/// The neuron holds its input edges in the network in the form of inputs, it maintains a separate
/// weight for each input edge, a shared bias, and an activation function.
#[derive(Debug)]
pub struct Neuron {
    /// 1-dimensional tensor of input values
    pub input: Tensor,
    /// 1-dimensional tensor of neuron weights, one for each input edge
    pub weights: Tensor,
    pub bias: f32,
    pub activation: &'static dyn ActivationFunction,
    pub output: f32,
    pub delta: f32,
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
            input: Tensor::zeros(input_size),
            weights: Tensor::uniform(input_size, -0.5, 0.5),
            bias: rand::random::<f32>() % 1.0,
            output: 0.0,
            delta: 0.0,
            activation,
        }
    }

    /// Make a forward pass through the neuron.
    pub fn forward(&mut self, input: &Tensor) -> f32 {
        let linear = vector::inner(input, &self.weights);
        self.output = self.activation.call(linear);
        self.output
    }

    /// Update the weights and bias of the neuron according to the given learning rate.
    pub fn update_weights(&mut self, learning_rate: f32) {
        vector::multiply(&mut self.weights, learning_rate);
        vector::multiply(&mut self.weights, self.delta);
        self.bias *= learning_rate * self.delta;
    }

    /// Manually update the bias value of the neuron.
    pub fn set_bias(&mut self, bias: f32) {
        self.bias = bias;
    }

    /// Manually update the weights of the neuron.
    pub fn set_weights(&mut self, weights: Tensor) {
        self.weights = weights;
    }
}
