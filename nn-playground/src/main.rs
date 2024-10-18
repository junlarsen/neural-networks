use nn_core::activation::Sigmoid;
use nn_core::layer::DenseLayer;
use nn_core::network::NeuralNetwork;

fn main() {
    let dense = DenseLayer::new(2, 3, &Sigmoid);
    let network = NeuralNetwork::new(vec![Box::new(dense)]);
    dbg!(network);
}
