use learning::Sequence;
use linealg::{Shape, Tensor, Vector};
use modules::{Linear, ReLU, Sigmoid, SqLoss, Translation};
use rand::{thread_rng, Rng};

mod learning;
mod linealg;
mod modules;

fn main() {
    let mut seq: Sequence = Sequence::new(
        Tensor::V(Vector(vec![0.0, 0.0])),
        vec![
            Box::new(Linear),
            Box::new(Translation),
            Box::new(Sigmoid),
            Box::new(Linear),
            Box::new(Translation),
            Box::new(Sigmoid),
            Box::new(Linear),
            Box::new(Translation),
            Box::new(Sigmoid),
            Box::new(SqLoss),
        ],
        vec![
            Tensor::zeroes(Shape::M(3, 2)),
            Tensor::zeroes(Shape::V(3)),
            Tensor::N,
            Tensor::zeroes(Shape::M(3, 3)),
            Tensor::zeroes(Shape::V(3)),
            Tensor::N,
            Tensor::zeroes(Shape::M(3, 3)),
            Tensor::zeroes(Shape::V(3)),
            Tensor::N,
            Tensor::zeroes(Shape::V(3)),
        ],
    );
    let lr = 0.1;
    for i in 0..100 {
        let x = thread_rng().gen_range(-2.0..25.);
        let y = x + 2.;
        seq.set_input(Tensor::V(Vector(vec![x, x])));
        seq.set_target(Tensor::V(Vector(vec![y, -y, 0.5 * y])));
        seq.forward();
        seq.backprop();
        seq.step(lr);

        if i % 10 == 0 {
            let (result, loss) = seq.get_result();
            println!("{result}, loss {loss}");
        }
    }
    println!("Done");
    for p in seq.get_params() {
        print!("{p}");
    }
}
