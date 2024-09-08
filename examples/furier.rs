use std::f32::consts::PI;

use hermostui::{
    learning::{count_loss, Optimizer, SGD},
    linealg::{Matrix, Tensor, Vector},
    modules::{FixParam, Function, Linear, MSELoss, Sequence, Sine, Translation},
};

#[allow(dead_code)]
fn sqare_wave(x: f32) -> f32 {
    if (x > 0.) == (x as i32 % 2 == 0) {
        1.0
    } else {
        0.0
    }
}

#[allow(dead_code)]
fn triangle_wave(x: f32) -> f32 {
    x.ceil() - x
}

#[allow(dead_code)]
fn weierstrass(x: f32) -> f32 {
    let a: f32 = 0.5;
    let b: f32 = 2.5;
    let mut sum = 0.0;
    for i in 1..5 {
        sum += a.powi(i) * (b.powi(i) * x * 2. * PI).sin();
    }
    sum
}

/// The function that model wants to learn.
fn signal(x: f32) -> f32 {
    //triangle_wave(x / 5.) * 5.
    //sqare_wave(x / 5.) * 6.
    //(x * 0.589 - 0.3499).sin()
    weierstrass(x / 4.0) * 8.
    //(x / 4.0).exp() - 8.
}

fn main() {
    // Number of waves, sine amount + cosine amount, better be even.
    let n_wave = 150;
    // Model's repeat unit, shall be same as input's interval's size.
    let period = 20.0f32;
    // Fixed frequency. Repeat once because of have to take care of cosine also.
    // 2pi / period * [1, 1, 2, 2, 3, 3, ...]
    let frequency_matrix = Tensor::M(Matrix::new(vec![Vector(
        (0..n_wave).map(|i| (i / 2 + 1) as f32 / period * 2. * PI).collect(),
    )]));
    // Every other is cosine.
    let phase_vector = Tensor::V(Vector(
        (0..n_wave)
            .map(|i| if i % 2 == 0 { 0.0 } else { PI / 2.0 })
            .collect(),
    ));
    let mut model = Sequence::new(vec![
        Box::new(FixParam::new(Linear::new(1, n_wave), frequency_matrix)),
        Box::new(FixParam::new(Translation::new(n_wave), phase_vector)),
        Box::new(Sine),
        Box::new(Linear::new(n_wave, 1)),
        Box::new(Translation::new(1)),
    ]);
    let mut param = Tensor::zeroes(model.param_shape());
    let inputs = (-100..=100).map(|i| Vector(vec![i as f32 / 10.]));
    let targets = inputs.clone().map(|x| Vector(vec![signal(x.0[0])]));
    let mut optim = SGD::new();
    let loss_fn = MSELoss;
    let lr = 1.0; // True, a rediculous big value, but it works! I don't know why :)
    for i in 0..30 {
        optim.step(
            &mut model,
            &mut param,
            &loss_fn,
            inputs.clone(),
            targets.clone(),
            lr,
        );

        if i % 2 == 0 {
            let loss = count_loss(&model, &param, &loss_fn, inputs.clone(), targets.clone());
            println!("epoch {i}, loss {loss}");
            // Uncomment if you want to see plot every other epoch, but then it's slow.
            //plot(&model, &param).unwrap();
        }
    }
    print!("{}", param);
    plot(&model, &param).unwrap();
    let loss = count_loss(&model, &param, &loss_fn, inputs.clone(), targets.clone());
    println!("Done, loss {loss}");
}

use plotters::prelude::*;
fn plot(model: &dyn Function, param: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("data/plot.bmp", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-10f32..10f32, -10f32..10f32)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (-1000..=1000)
                .map(|x| x as f32 / 100.0)
                .map(|x| (x, signal(x))),
            &BLUE,
        ))?
        .label("Aproching function")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            (-1000..=1000)
                .map(|x| x as f32 / 100.0)
                .map(|x| (x, model.evaluate(param, &Vector(vec![x])).0[0])),
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
