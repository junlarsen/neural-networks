//! Neural Network layers

use crate::activation::ActivationFunction;
use crate::neuron::Neuron;
use crate::tensor::{Tensor, Tensor1D};
use std::fmt::Debug;

pub trait Layer: Debug {
    fn forward(&mut self, input: Tensor1D) -> Tensor1D;
    fn backward(&mut self, previous: &Vec<Neuron>, example: f32, is_output_layer: bool);
}

#[derive(Debug)]
pub struct DenseLayer {
    neurons: Vec<Neuron>,
}

impl Layer for DenseLayer {
    /// Pass the input through each of the neurons in the network
    fn forward(&mut self, input: Tensor1D) -> Tensor1D {
        let outputs = self
            .neurons
            .iter_mut()
            .map(|n| n.forward(&input))
            .collect::<Vec<_>>();
        Tensor1D::from_vec(outputs)
    }

    fn backward(&mut self, previous: &Vec<Neuron>, example: f32, is_output_layer: bool) {
        unimplemented!()
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
