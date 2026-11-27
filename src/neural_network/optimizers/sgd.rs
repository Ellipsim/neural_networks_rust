
use crate::neural_network::gradients::NetworkGradients;
use crate::neural_network::network::Network;
use crate::neural_network::optimizers::Optimizer;

pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(
        &mut self,
        network: &mut Network,
        gradients: &NetworkGradients,
    ) {
        network.update_parameters(gradients, self.learning_rate);
    }
}
