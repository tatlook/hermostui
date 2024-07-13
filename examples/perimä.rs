use std::fs;
use std::io::Write;
use std::iter::zip;

use hermostui::{
    learning::SGD,
    linealg::{Tensor, Vector},
    modules::{Function, Linear, LossFunction, MSELoss, ReLU, Sequence, Translation},
};

fn approchfun(x: f32) -> f32 {
    (x / 2.).cos()
}

fn main() {
    let model = Sequence::new(vec![
        Box::new(Linear::new(1, 1)),
        Box::new(Translation::new(1)),
        Box::new(ReLU),
        Box::new(Linear::new(1, 1)),
        Box::new(Translation::new(1)),
    ]);
    let param = Tensor::rand(model.param_shape());
    let mut optim = SGD::new(
        Box::new(model),
        param,
        Box::new(MSELoss),
    );
    if let Ok(data) = fs::read("data/perimä.pkl") {
        println!("Resuming training from data/perimä.pkl");
        let params: Tensor = serde_pickle::from_slice(&data, Default::default()).unwrap();
        optim.set_param(params);
    } else {
        println!("Training from scratch");
    }
    let lr = 1e-1;
    for i in 0..10000 {
        let inputs: Vec<Vector> = (0..100)
            .map(|_| Vector(vec![rand::thread_rng().gen_range(-10.0..10.0)]))
            .collect();
        let targets: Vec<Vector> = inputs
            .iter()
            .map(|input| Vector(vec![approchfun(input.0[0])]))
            .collect();
        let mut loss_sum_before = 0.0;
        for (input, target) in zip(&inputs, &targets) {
            let value = optim.evaluate(input.clone());
            let loss = MSELoss.loss(target, &value);
            loss_sum_before += loss;
        }
        let params_before = optim.get_param();
        let params_after = Tensor::rand(params_before.shape()) * lr + &params_before;

        optim.set_param(params_after);

        let mut loss_sum_after = 0.0;
        for (input, target) in zip(&inputs, &targets) {
            let value = optim.evaluate(input.clone());
            let loss = MSELoss.loss(target, &value);
            loss_sum_after += loss;
        }
        if loss_sum_after > loss_sum_before {
            optim.set_param(params_before);
        }

        if i % 1000 == 0 {
            println!("epoch {i}, loss {loss_sum_before}");
            if loss_sum_before.is_nan() {
                println!("loss is nan, something went wrong, exit");
                return;
            }
        }

        if i % 3000 == 2999 {
            println!("Saving params");
            let binary = serde_pickle::to_vec(&optim.get_param(), Default::default()).unwrap();
            if let Ok(mut file) = std::fs::File::create("data/perimä.pkl") {
                file.write_all(&binary).unwrap();
            } else {
                println!("Failed to save params");
            }
            plot(&mut optim).unwrap();
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
