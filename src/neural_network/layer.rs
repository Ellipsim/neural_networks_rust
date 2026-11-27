
use ndarray::Array2;

#[derive(Clone, Debug)]
pub struct Layer {
    // Shape: (input_size, output_size)
    pub weights: Array2<f32>,

    // Shape: (1, output_size)
    pub biases: Array2<f32>,
}

impl Layer {
    pub fn new(
        weights: Array2<f32>,
        biases: Array2<f32>,
    ) -> Self {
        assert!(weights.nrows() > 0);
        assert!(weights.ncols() > 0);
        assert_eq!(biases.dim(), (1, weights.ncols()));

        Self { weights, biases }
    }

    pub fn input_size(&self) -> usize {
        self.weights.nrows()
    }

    pub fn output_size(&self) -> usize {
        self.weights.ncols()
    }
}
