//! Neural Network layers

use crate::activation::ActivationFunction;
use crate::linalg::Vector;
use crate::neuron::Neuron;

pub trait Layer {}

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
        let mut neurons = Vec::with_capacity(neuron_count);
        for _ in 0..neuron_count {
            neurons.push(Neuron::new(input_size, activation));
        }
        Self { neurons }
    }

    /// Pass the input through each of the neurons in the network
    pub fn forward(&mut self, input: Vector) -> Vector {
        let mut output = Vector::new(vec![0.0; self.neurons.len()]);
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            output.set(i, neuron.forward(&input));
        }
        output
    }
}
