use std::fs;
use std::io::Write;

use hermostui::{
    learning::SGD,
    linealg::{Shape, Tensor, Vector},
    modules::{Linear, MSELoss, ReLU, Sequence, Translation},
};

fn approchfun(x: f32) -> f32 {
    (x * 2.0).sin() * 6.0 - 1.0
}

fn main() {
    let mut seq: SGD = SGD::new(
        vec![Box::new(Sequence::new(vec![
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
            Box::new(ReLU),
            Box::new(Linear),
            Box::new(Translation),
        ]))],
        vec![Tensor::L(vec![
            Tensor::rand(Shape::M(16, 1)),
            Tensor::rand(Shape::V(16)),
            Tensor::N,
            Tensor::rand(Shape::M(16, 16)),
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
        ])],
        Box::new(MSELoss),
    );
    if let Ok(data) = fs::read("data/sin_approch.pkl") {
        println!("Resuming training from data/sin_approch.pkl");
        let params: Vec<Tensor> = serde_pickle::from_slice(&data, Default::default()).unwrap();
        seq.set_params(params);
    } else {
        println!("Training from scratch");
    }
    let lr = 1e-3;
    for i in 0..10000 {
        seq.zero_grad();
        for _ in 0..10 {
            let x = rand::thread_rng().gen_range(-10.0..10.0);
            let y = approchfun(x);
            seq.forward(Vector(vec![x]));
            seq.backprop(Vector(vec![y]));
        }
        seq.step(lr);

        if i % 1000 == 0 {
            let loss = seq.get_loss();
            println!("epoch {i}, loss {loss}");
            if loss.is_nan() {
                println!("loss is nan, something went wrong, exit");
                return;
            }
        }

        if i % 3000 == 2999 {
            println!("Saving params");
            let binary = serde_pickle::to_vec(&seq.get_params(), Default::default()).unwrap();
            if let Ok(mut file) = std::fs::File::create("data/sin_approch.pkl") {
                file.write_all(&binary).unwrap();
            } else {
                println!("Failed to save params");
            }
            plot(&mut seq).unwrap();
        }
    }
    println!("Done");
}

use plotters::prelude::*;
use rand::Rng;
fn plot(seq: &mut SGD) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("data/plot.bmp", (640, 480)).into_drawing_area();
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
            (-1000..=1000)
                .map(|x| x as f32 / 100.0)
                .map(|x| (x, seq.evaluate(Vector(vec![x])).0[0])),
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
