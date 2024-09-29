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
    // Actual gradient is param_grad / batch_size
    // Should always use via take_gradient()
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
            debug_assert_eq!(self.batch_size, 0);
            self.param_grad = Some(grad);
        }

        self.batch_size += 1;
    }
    fn take_gradient(&mut self) -> Tensor {
        let grad = self.param_grad.take().unwrap() * (1.0 / self.batch_size as f32);
        self.batch_size = 0;
        grad
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
        let param_grad = self.take_gradient();
        *param -= &(param_grad * lr);
    }
}

pub struct Adam {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    sgd: SGD,
    t: i32,
    moment_1: Tensor,
    moment_2: Tensor,
}

impl Adam {
    pub fn new(beta_1: f32, beta_2: f32, epsilon: f32) -> Self {
        assert!(0.0 < beta_1 && beta_1 < 1.0, "beta_1 must be in (0, 1)");
        assert!(0.0 < beta_2 && beta_2 < 1.0, "beta_2 must be in (0, 1)");
        assert!(epsilon > 0.0, "epsilon must be positive");
        Self {
            beta_1,
            beta_2,
            epsilon,
            sgd: SGD::new(),
            t: 0,
            moment_1: Tensor::N,
            moment_2: Tensor::N,
        }
    }
}

impl Default for Adam {
    fn default() -> Self {
        Adam::new(0.9, 0.999, 1e-8)
    }
}

impl Optimizer for Adam {
    fn step(
        &mut self,
        func: &mut dyn Function,
        param: &mut Tensor,
        loss_fn: &dyn LossFunction,
        inputs: impl Iterator<Item = Vector> + Clone,
        targets: impl Iterator<Item = Vector> + Clone,
        lr: f32,
    ) {
        if self.t == 0 {
            self.moment_1 = Tensor::zeroes(param.shape());
            self.moment_2 = Tensor::zeroes(param.shape());
        }
        self.t += 1;
        self.sgd.eat_batch(func, param, loss_fn, inputs, targets);
        let mut grad = self.sgd.take_gradient();

        self.moment_1 *= self.beta_1;
        self.moment_1 += &(grad.clone() * (1.0 - self.beta_1));
        self.moment_2 *= self.beta_2;
        grad.iterover_assign(&|x| x * x * (1.0 - self.beta_2));
        self.moment_2 += &grad;
        let lr = lr * (1.0 - self.beta_2.powi(self.t)).sqrt() / (1.0 - self.beta_1.powi(self.t));
        let mut correct_2 = self.moment_2.clone();
        correct_2.iterover_assign(&|x| 1.0 / (x.sqrt() + self.epsilon) * lr);
        *param -= &correct_2.hadamard(&self.moment_1);
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
