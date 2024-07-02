use std::{fs, io::Write};

use learning::Sequence;
use linealg::{Shape, Tensor, Vector};
use modules::{Linear, MSELoss, ReLU, Translation};
use rand::{thread_rng, Rng};

mod learning;
mod linealg;
mod modules;

fn approchfun(x: f32) -> f32 {
    (x * 2.0).sin() * 2.0 - 1.0
    // fn sigmoid(x:f32) -> f32 {

    //     1.0 / (1.0 + (-x).exp())
    // }

    // sigmoid(x * 0.5 + 3.) * 2.
}

fn main() {
    let mut seq: Sequence = Sequence::new(
        vec![
            Box::new(Linear),
            Box::new(Translation),
            Box::new(ReLU),
            Box::new(Linear),
            Box::new(Translation),
            Box::new(ReLU),
            Box::new(Linear),
            Box::new(Translation),
            Box::new(ReLU),
            Box::new(Linear),
            Box::new(Translation),
        ],
        vec![
            Tensor::rand(Shape::M(16, 1)),
            Tensor::rand(Shape::V(16)),
            Tensor::N,
            Tensor::rand(Shape::M(16, 16)),
            Tensor::rand(Shape::V(16)),
            Tensor::N,
            Tensor::rand(Shape::M(16, 16)),
            Tensor::rand(Shape::V(16)),
            Tensor::N,
            Tensor::rand(Shape::M(1, 16)),
            Tensor::rand(Shape::V(1)),
        ],
        Box::new(MSELoss),
    );
    if let Ok(data) = fs::read("params.pkl") {
        println!("Resuming training from params.pkl");
        let params: Vec<Tensor> = serde_pickle::from_slice(&data, Default::default()).unwrap();
        seq.set_params(params);
    } else {
        println!("Training from scratch");
    }
    let lr = 4e-3;
    for i in 0..10000 {
        seq.zero_grad();
        for _ in 0..10 {
            let x = thread_rng().gen_range(-10.0..10.);
            let y = approchfun(x);
            seq.set_input(Tensor::V(Vector(vec![x])));
            seq.set_target(Tensor::V(Vector(vec![y])));
            seq.forward();
            seq.backprop();
        }
        seq.step(lr);

        if i % 1000 == 0 {
            let (_, loss) = seq.get_result();
            println!("epoch {i}, loss {loss}");
        }
    }
    println!("Done");
    for p in seq.get_params() {
        print!("{p}");
    }
    let binary = serde_pickle::to_vec(&seq.get_params(), Default::default()).unwrap();
    if let Ok(mut file) = std::fs::File::create("params.pkl") {
        file.write_all(&binary).unwrap();
    }
    maisak(seq).unwrap();
}

use plotters::prelude::*;
fn maisak(mut seq: Sequence) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plot.bmp", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-10f32..10f32, -10f32..10f32)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (-1000..=1000)
                .map(|x| x as f32 / 100.0)
                .map(|x| (x, approchfun(x))),
            &BLUE,
        ))?
        .label("Aproching function")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            (-1000..=1000).map(|x| x as f32 / 100.0).map(|x| {
                seq.set_input(Tensor::V(Vector(vec![x])));
                seq.forward();
                (x, seq.get_result().0.as_vector().0[0])
            }),
            &RED,
        ))?
        .label("Network")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
