use dady::{Constant, Function, SqLoss};
use linealg::{Tensor, Vector};
use rand::{thread_rng, Rng};

//mod nn;
mod dady;
mod linealg;

fn fntoapproch(x: f32) -> f32 {
    3. + 2.0 * x
}

fn optimize0(func: &mut dyn Function, mut delta: Tensor, lr: f32) {
    if let None = func.input() {
        return
    }
    let (nabla_param, nabla_in) = func.gradient();
    delta *= nabla_param;
    let param = func.params_mut();
    *param -= delta.clone() * Tensor::S(lr);
    
    optimize0(func.input_mut().unwrap(), delta, lr);
}

fn optimize(func: &mut dyn Function, lr: f32) {
    let (_, nabla_in) = func.gradient();
    let delta = nabla_in;
    optimize0(func, delta, lr);
}

fn set_input(func: &mut dyn Function, x: Tensor) {
    if let Some(mut input) = func.input_mut() {
        set_input(input, x);
    } else {
        func.params_mut().set(x);
    }
}

fn main() {
    let fs = dady::Polynomial::new(
        Vector(vec![3., -2., 1.0]),
        Box::new(dady::Polynomial::new(
            Vector(vec![0.0, 1.0]),
            Box::new(Constant(Tensor::S(1.0))),
        )),
    );
    let mut loss = SqLoss::new(Box::new(fs), Tensor::S(0.5));
    let lr = 0.1;
    for _ in 0..1000 {
        let x = thread_rng().gen_range(-1.0..1.0);
        let y = fntoapproch(x);
        set_input(&mut loss, Tensor::S(x));
        loss.params_mut().set(Tensor::S(y));
        optimize(&mut loss, lr);
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn reduce() {
        let v = vec![1., 2., 4., 5.];
        assert_eq!(Vec::from(&v[..]), v);
        assert_eq!(Vec::from(&v[..v.len()]), v);
        assert_eq!(Vec::from(&v[..v.len() - 1]), vec![1., 2., 4.]);
    }
}
