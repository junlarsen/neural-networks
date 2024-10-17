use nn_core::activation::Sigmoid;
use nn_core::layer::DenseLayer;
use nn_core::network::NeuralNetwork;

fn main() {
    let _ = NeuralNetwork::new(vec![
        &DenseLayer::new(2, 3, &Sigmoid)
    ]);
}
