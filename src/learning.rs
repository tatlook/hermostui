use std::iter::zip;

use crate::{
    linealg::{Tensor, Vector},
    modules::{Function, LossFunction},
};

pub struct Sequence {
    input: Option<Vector>,
    funcs: Vec<Box<dyn Function>>,
    params: Vec<Tensor>,
    loss_fn: Box<dyn LossFunction>,
    loss: f32,
    target: Option<Vector>,
    output_chache: Vec<Vector>,
    param_grads: Option<Vec<Tensor>>,
    batch_size: usize,
}

impl Sequence {
    pub fn new(
        funcs: Vec<Box<dyn Function>>,
        params: Vec<Tensor>,
        loss_fn: Box<dyn LossFunction>,
    ) -> Self {
        Self {
            input: None,
            funcs,
            params,
            loss_fn,
            loss: 0.0,
            target: None,
            output_chache: vec![],
            param_grads: None,
            batch_size: 0,
        }
    }

    pub fn set_input(&mut self, input: Vector) {
        self.input = Some(input);
        self.output_chache.clear();
    }

    pub fn set_target(&mut self, target: Vector) {
        self.target = Some(target);
    }

    pub fn forward(&mut self) {
        self.output_chache.clear();
        let mut input = self.input.clone().unwrap();

        for (func, param) in zip(&self.funcs, &self.params) {
            let output = func.forward(param, &input);
            self.output_chache.push(output.clone());
            input = output;
        }

        self.loss += self.loss_fn.loss(&self.target.clone().unwrap(), &input);
    }

    /// Only evaluate, returns output on last layer
    pub fn evaluate(&self) -> Vector {
        let mut input = self.input.clone().unwrap();

        for (func, param) in zip(&self.funcs, &self.params) {
            let output = func.forward(param, &input);
            input = output;
        }

        input
    }
    
    pub fn backprop(&mut self) {
        let mut param_grads = vec![];
        let mut delta = self
            .loss_fn
            .delta(&self.target.clone().unwrap(), &self.output_chache.last().unwrap());
        for i in 0..self.funcs.len() {
            let i = self.funcs.len() - 1 - i;
            let func = &self.funcs[i];
            let param = &self.params[i];
            let input = if i == 0 {
                self.input.clone().unwrap()
            } else {
                self.output_chache[i - 1].clone()
            };
            let (grad, new_delta) = func.backward(param, &input, &delta);
            delta = new_delta;
            param_grads.insert(0, grad);
        }
        
        if let Some(param_grads_sum) = &mut self.param_grads {
            for i in 0..param_grads_sum.len() {
                param_grads_sum[i] += param_grads[i].clone();
            }
        } else {
            assert!(self.batch_size == 0);
            self.param_grads = Some(param_grads);
        }
        self.batch_size += 1;
    }
    
    pub fn step(&mut self, lr: f32) {
        if let Some(param_grads) = &self.param_grads {
            for i in 0..self.funcs.len() {
                let param_grad = param_grads[i].clone();
                let param = &mut self.params[i];
                *param -= Tensor::S(lr / self.batch_size as f32) * param_grad;
            }
        } else {
            panic!("No gradient to desent")
        }
    }

    pub fn zero_grad(&mut self) {
        self.param_grads = None;
        self.batch_size = 0;
        self.loss = 0.0;
    } 

    pub fn get_params(&self) -> Vec<Tensor> {
        self.params.clone()
    }

    pub fn set_params(&mut self, params: Vec<Tensor>) {
        self.params = params;
        self.zero_grad();
    }

    /// Returns (network output, loss)
    pub fn get_result(&self) -> (Vector, f32) {
        let output = self.output_chache.last().unwrap().clone();
        (output, self.loss / self.batch_size as f32)
    }
}
