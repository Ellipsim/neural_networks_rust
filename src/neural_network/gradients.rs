
use ndarray::Array2;

#[derive(Clone, Debug)]
pub struct LayerGradients {
    pub weights: Array2<f32>,
    pub biases: Array2<f32>,
}

#[derive(Clone, Debug)]
pub struct NetworkGradients {
    pub layers: Vec<LayerGradients>,
}

impl NetworkGradients {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        assert_eq!(self.layers.len(), other.layers.len());

        for (layer, other_layer) in
            self.layers.iter_mut().zip(&other.layers)
        {
            layer.weights += &other_layer.weights;
            layer.biases += &other_layer.biases;
        }
    }

    pub fn scale(&mut self, factor: f32) {
        for layer in &mut self.layers {
            layer.weights.mapv_inplace(|x| x * factor);
            layer.biases.mapv_inplace(|x| x * factor);
        }
    }
}

impl Default for NetworkGradients {
    fn default() -> Self {
        Self::new()
    }
}
