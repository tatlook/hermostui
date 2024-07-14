use std::iter::zip;

use crate::{
    linealg::{Tensor, Vector},
    modules::{Function, LossFunction},
};

pub struct SGD {
    param_grad: Option<Tensor>,
    batch_size: usize,
}

impl SGD {
    pub fn new() -> Self {
        Self {
            param_grad: None,
            batch_size: 0,
        }
    }
}

impl SGD {
    pub fn eat_batch(
        &mut self,
        func: &mut dyn Function,
        param: &Tensor,
        loss_fn: &dyn LossFunction,
        inputs: impl Iterator<Item = Vector>,
        targets: impl Iterator<Item = Vector>,
    ) {
        for (input, target) in zip(inputs, targets) {
            self.eat_sample(func, param, loss_fn, input, target);
        }
    }
    pub fn eat_sample(
        &mut self,
        func: &mut dyn Function,
        param: &Tensor,
        loss_fn: &dyn LossFunction,
        input: Vector,
        target: Vector,
    ) {
        let output = func.forward(param, &input);
        let delta = loss_fn.delta(&target, &output);
        let (grad, _) = func.backward(param, &input, delta);

        if let Some(param_grad_sum) = &mut self.param_grad {
            *param_grad_sum += &grad;
        } else {
            assert!(self.batch_size == 0);
            self.param_grad = Some(grad);
        }

        self.batch_size += 1;
    }

    pub fn step(&mut self, param: &mut Tensor, lr: f32) {
        if let Some(param_grad) = self.param_grad.take() {
            *param -= &(param_grad * (lr / self.batch_size as f32));
        } else {
            panic!("No gradient to desent")
        }
        self.batch_size = 0;
    }
}

pub struct TryStep;

impl TryStep {
    pub fn step(
        &self,
        func: &dyn Function,
        param: &mut Tensor,
        loss_fn: &dyn LossFunction,
        inputs: impl Iterator<Item = Vector> + Clone,
        targets: impl Iterator<Item = Vector> + Clone,
        lr: f32,
    ) {
        let loss = count_loss(func, param, loss_fn, inputs.clone(), targets.clone());
        let param_after = Tensor::rand(param.shape()) * lr + param;
        let loss_after = count_loss(func, &param_after, loss_fn, inputs, targets);

        if loss_after < loss {
            *param = param_after;
        }
    }
}

pub fn count_loss(
    func: &dyn Function,
    param: &Tensor,
    loss_fn: &dyn LossFunction,
    inputs: impl Iterator<Item = Vector>,
    targets: impl Iterator<Item = Vector>,
) -> f32 {
    let mut loss = 0.0;
    let mut n = 0;
    for (input, target) in zip(inputs, targets) {
        let output = func.evaluate(param, &input);
        loss += loss_fn.loss(&target, &output);
        n += 1;
    }
    loss / n as f32
}
