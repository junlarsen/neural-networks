//! Representation of a neuron in a neural network.
//!
//! This module's implementation of a neuron is rather simplistic. It has a number of input
//! parameters, each having a weight, a single bias, and an activation function associated with it.
//!
//! The neuron also provides an output value, a delta value for networks that use backpropagation,
//! and a method to update the weights based on the delta value.

use crate::activation::ActivationFunction;
use crate::linalg::Vector;

/// A single neuron in a neural network.
///
/// The neuron holds its input edges in the network in the form of inputs, it maintains a separate
/// weight for each input edge, a shared bias, and an activation function.
pub struct Neuron {
    pub input: Vector,
    pub weights: Vector,
    pub bias: f32,
    pub activation: Box<dyn ActivationFunction>,
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
    pub fn new(input_size: usize, activation: impl ActivationFunction + 'static) -> Self {
        Self {
            input: Vector::new(vec![0.0; input_size]),
            weights: Vector::from_uniform_distribution(input_size + 1, -0.5, 0.5),
            activation: Box::new(activation),
            bias: 0.0,
            output: 0.0,
            delta: 0.0,
        }
    }

    /// Make a forward pass through the neuron.
    pub fn forward(&mut self, input: Vector) -> f32 {
        let linear = input.inner_product(&self.weights);
        self.output = self.activation.call(linear);
        self.output
    }

    /// Manually update the bias value of the neuron.
    pub fn set_bias(&mut self, bias: f32) {
        self.bias = bias;
    }

    /// Manually update the weights of the neuron.
    pub fn set_weights(&mut self, weights: Vector) {
        self.weights = weights;
    }
}
