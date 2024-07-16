use std::iter::zip;

use crate::{
    linealg::{Tensor, Vector},
    modules::{Function, LossFunction},
};

pub trait Optimizer {
    fn step(
        &mut self,
        func: &mut dyn Function,
        param: &mut Tensor,
        loss_fn: &dyn LossFunction,
        inputs: impl Iterator<Item = Vector> + Clone,
        targets: impl Iterator<Item = Vector> + Clone,
        lr: f32,
    );
}

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
    fn eat_batch(
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
    fn eat_sample(
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
}

impl Optimizer for SGD {
    fn step(
        &mut self,
        func: &mut dyn Function,
        param: &mut Tensor,
        loss_fn: &dyn LossFunction,
        inputs: impl Iterator<Item = Vector> + Clone,
        targets: impl Iterator<Item = Vector> + Clone,
        lr: f32,
    ) {
        self.eat_batch(func, param, loss_fn, inputs, targets);

        if let Some(param_grad) = self.param_grad.take() {
            *param -= &(param_grad * (lr / self.batch_size as f32));
        } else {
            unreachable!()
        }
        self.batch_size = 0;
    }
}

pub struct TryStep;

impl Optimizer for TryStep {
    fn step(
        &mut self,
        func: &mut dyn Function,
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

pub struct Genetic {
    n_spices: usize,
    n_survival: usize,
    spices: Vec<(Tensor, f32)>,
    /// Best spices's clone, don't mutate during mutation.
    /// Will join other spices in natural selection, if it's children are getting worse.
    champion: (Tensor, f32),
}

impl Genetic {
    pub fn new(initial: Tensor, n_spices: usize, n_survival: usize) -> Self {
        assert!(n_spices >= n_survival);
        let mut spices: Vec<(Tensor, f32)> = vec![(Tensor::N, f32::default()); n_spices];
        for (param, _) in &mut spices {
            *param = Tensor::rand(initial.shape());
        }
        spices[0].0 = initial.clone();
        Self {
            n_spices,
            n_survival,
            spices,
            champion: (initial, f32::MAX),
        }
    }
    fn natural_selection(&mut self) {
        self.spices
            .sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        self.spices.truncate(self.n_survival);
        for i in 0..self.n_spices - self.n_survival {
            self.spices.push(self.spices[i % self.n_survival].clone());
        }
        // Make sure after mutation the thing don't get worse
        if self.champion.1 < self.spices[0].1 {
            self.spices[0] = self.champion.clone();
        } else {
            // Now we have a new champion
            self.champion = self.spices[0].clone();
        }
    }
    fn random_mutate(&mut self, lr: f32) {
        for (param, _) in &mut self.spices {
            *param += &(Tensor::rand(param.shape()) * lr);
        }
    }
    fn update_losses(
        &mut self,
        func: &dyn Function,
        loss_fn: &dyn LossFunction,
        inputs: impl Iterator<Item = Vector> + Clone,
        targets: impl Iterator<Item = Vector> + Clone,
    ) {
        for (param, loss) in &mut self.spices {
            *loss = count_loss(func, param, loss_fn, inputs.clone(), targets.clone());
        }
    }
}

impl Optimizer for Genetic {
    fn step(
        &mut self,
        func: &mut dyn Function,
        param: &mut Tensor,
        loss_fn: &dyn LossFunction,
        inputs: impl Iterator<Item = Vector> + Clone,
        targets: impl Iterator<Item = Vector> + Clone,
        lr: f32,
    ) {
        self.random_mutate(lr);
        self.update_losses(func, loss_fn, inputs, targets);
        self.natural_selection();
        *param = self.champion.0.clone();
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
