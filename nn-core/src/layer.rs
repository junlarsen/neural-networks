//! Neural Network layers

use crate::activation::ActivationFunction;
use crate::neuron::Neuron;
use crate::tensor::Tensor;
use std::fmt::Debug;

pub trait Layer: Debug {}

#[derive(Debug)]
pub struct DenseLayer {
    neurons: Vec<Neuron>,
}

impl Layer for DenseLayer {}

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

    /// Pass the input through each of the neurons in the network
    pub fn forward(&mut self, input: Tensor) -> Tensor {
        let outputs = self
            .neurons
            .iter_mut()
            .map(|n| n.forward(&input))
            .collect::<Vec<_>>();
        Tensor::vector(outputs)
    }
}
