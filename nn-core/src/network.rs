use crate::layer::Layer;

pub struct NeuralNetwork<'a> {
    pub layers: Vec<&'a dyn Layer>
}

impl<'a> NeuralNetwork<'a> {
    pub fn new(layers: Vec<&'a dyn Layer>) -> Self {
        Self { layers }
    }
}
