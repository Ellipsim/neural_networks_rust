
use crate::neural_network::gradients::NetworkGradients;
use crate::neural_network::network::Network;

pub mod adam;
pub mod sgd;

pub trait Optimizer {
    fn update(
        &mut self,
        network: &mut Network,
        gradients: &NetworkGradients,
    );
}
