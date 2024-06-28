use crate::linealg::Vector;

pub trait F32Func {
    fn call(&self, x: f32) -> f32;
    fn derivative(&self, x: f32) -> f32;
}

pub trait VectorFunc {
    fn call(&self, x: &Vector) -> Vector;
    fn derivative(&self, x: &Vector) -> Vector;
}

pub struct Sqare {
}

impl F32Func for Sqare {
    fn call(&self, x: f32) -> f32 {
        x * x
    }
    fn derivative(&self, x: f32) -> f32 {
        2. * x
    }
}

pub struct ReLU {
}

impl F32Func for ReLU {
    fn call(&self, x: f32) -> f32 {
        x.max(0.0)
    }
    fn derivative(&self, x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }    
    }
}

pub struct ForAllElements(pub Box<dyn F32Func>);

impl VectorFunc for ForAllElements {
    fn call(&self, x: &Vector) -> Vector {
        Vector(x.iter().map(|x| self.0.call(*x)).collect())
    }
    fn derivative(&self, x: &Vector) -> Vector {
        Vector(x.iter().map(|x| self.0.derivative(*x)).collect())
    }
}

pub struct Linear {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vector>,
    bias: Vec<f32>,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize, weights: Vec<Vector>, bias: Vec<f32>) -> Self {
        assert_eq!(weights.len(), output_size);
        assert_eq!(bias.len(), output_size);
        Self { input_size, output_size, weights, bias }
    }
}

impl VectorFunc for Linear {
    fn call(&self, x: &Vector) -> Vector {
        assert_eq!(x.size(), self.input_size);
        let mut y = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            y[i] += self.weights[i].dot(&x);
            y[i] += self.bias[i];
        }
        Vector(y)
    }
    fn derivative(&self, x: &Vector) -> Vector {
        assert_eq!(x.size(), self.input_size);
        let mut y = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            y[i] += self.weights[i].len_sq(); // TODO:
        }
        Vector(y)
    }
}
