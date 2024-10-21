//! Neural Network layers

use crate::activation::ActivationFunction;
use crate::neuron::Neuron;
use crate::tensor::{Tensor1D, Tensor2D};
use std::fmt::Debug;

pub trait Layer: Debug {
    fn forward(&mut self, input: Tensor2D) -> Tensor1D;
    fn backward(&mut self, output_error: Tensor1D, learning_rate: f32);
}

#[derive(Debug)]
pub struct DenseLayer {
    neurons: Vec<Neuron>,
    activation: &'static dyn ActivationFunction,
    /// The input values to the layer, stored as a 2-dimensional matrix
    ///
    /// This matrix has the shape (number of neurons, number of edges from the previous layer)
    input: Tensor2D,
    weights: Tensor2D,
    /// The output values of the layer, represented the same way as the input
    output: Tensor2D,
    /// The bias values for each of the neurons in the layer
    bias: Tensor1D,
}

impl Layer for DenseLayer {
    /// Pass the input through each of the neurons in the network
    fn forward(&mut self, input: Tensor2D) -> Tensor2D {
        self.input = input;
        
    }

    fn backward(&mut self, output_error: Tensor1D, learning_rate: f32) {
        let weights = self.neurons.iter().map(|n| n.weights().inner())
            .flatten()
            .collect::<Vec<_>>();
        let rows = self.neurons.len();
        let cols = self.neurons.iter().map(|n| n.weights().length()).sum();
        let weights = Tensor2D::from_raw(weights, [rows, cols]);
        let output_error = weights.dot(&output_error.transpose());

        todo!()
        // let error = output_error.pairwise_multiplication()
    }
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        neuron_count: usize,
        activation: &'static impl ActivationFunction,
    ) -> Self {
        let neurons = (0..neuron_count)
            .map(|_| Neuron::new(input_size, activation))
            .collect();
        Self { neurons }
    }

    /// Update the weights of each neuron in the layer
    pub fn update_weights(&mut self, learning_rate: f32) {
        for neuron in self.neurons.iter_mut() {
            neuron.update_weights(learning_rate);
        }
    }
}
