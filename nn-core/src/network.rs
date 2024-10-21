use crate::layer::Layer;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }
}
