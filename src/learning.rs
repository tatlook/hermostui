use std::iter::zip;

use crate::{linealg::Tensor, modules::Function};

pub struct Sequence {
    funcs: Vec<Box<dyn Function>>,
    output_chache: Vec<Tensor>,
    input: Tensor,
    params: Vec<Tensor>,
    param_grads: Vec<Tensor>,
}

impl Sequence {
    pub fn new(input: Tensor, funcs: Vec<Box<dyn Function>>, params: Vec<Tensor>) -> Self {
        Self {
            funcs,
            output_chache: vec![],
            input,
            params,
            param_grads: vec![],
        }
    }

    pub fn set_input(&mut self, input: Tensor) {
        self.input = input;
        self.output_chache.clear();
    }

    pub fn set_target(&mut self, target: Tensor) {
        *self.params.last_mut().unwrap() = target;
    }

    pub fn forward(&mut self) {
        self.output_chache.clear();
        let mut input = self.input.clone();

        for (func, param) in zip(&self.funcs, &self.params) {
            let output = func.forward(param, &input);
            self.output_chache.push(output.clone());
            input = output;
        }
    }

    pub fn backprop(&mut self) {
        self.param_grads.clear();
        let (_, mut delta) = self.funcs.last().unwrap().backward(
            self.params.last().unwrap(),
            &self.output_chache[self.output_chache.len() - 2],
            &Tensor::S(0.0),
        );
        for i in 0..self.funcs.len() - 1 {
            let i = self.funcs.len() - 2 - i;
            let func = &self.funcs[i];
            let param = &self.params[i];
            let input = if i == 0 {
                &self.input
            } else {
                &self.output_chache[i - 1]
            };
            let (grad, new_delta) = func.backward(param, input, &delta);
            delta = new_delta;
            self.param_grads.insert(0, grad);
        }
    }

    pub fn step(&mut self, lr: f32) {
        for i in 0..self.funcs.len() - 1 {
            let param_grad = self.param_grads[i].clone();
            let param = &mut self.params[i];
            *param -= Tensor::S(lr) * param_grad;
        }
    }

    pub fn get_params(&self) -> Vec<Tensor> {
        self.params.clone()
    }

    /// Returns (network output, loss)
    pub fn get_result(&self) -> (Tensor, f32) {
        let output = self.output_chache[self.output_chache.len() - 2].clone();
        let loss = self.output_chache.last().unwrap().as_scalar();
        (output, loss)
    }
}
