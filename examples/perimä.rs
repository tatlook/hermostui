use std::fs;

use hermostui::{
    learning::{count_loss, Genetic, Optimizer},
    linealg::{Tensor, Vector},
    modules::{Function, Linear, MSELoss, ReLU, Sequence, Translation},
};

fn approchfun(x: f32) -> f32 {
    (x / 1.).cos() * 3.
}

fn main() {
    let mut model = Sequence::new(vec![
        Box::new(Linear::new(1, 10)),
        Box::new(Translation::new(10)),
        Box::new(ReLU),
        Box::new(Linear::new(10, 1)),
        Box::new(Translation::new(1)),
    ]);
    let mut param = if let Ok(data) = fs::read("data/perimä.pkl") {
        println!("Resuming training from data/perimä.pkl");
        serde_pickle::from_slice(&data, Default::default()).unwrap()
    } else {
        println!("Training from scratch");
        Tensor::rand(model.param_shape())
    };
    let mut optim = Genetic::new(param.clone(), 20, 5, );
    let mut lr = 1e-1;
    for i in 0..10000 {
        let inputs = (0..50).map(|i| Vector(vec![i as f32 / 2.5 - 10.0]));
        let targets = inputs.clone().map(|x| Vector(vec![approchfun(x.0[0])]));
        optim.step(
            &mut model,
            &mut param,
            &MSELoss,
            inputs.clone(),
            targets.clone(),
            lr,
        );

        if i % 300 == 0 {
            let loss = count_loss(&model, &param, &MSELoss, inputs, targets);
            if loss < 1.0 {
                lr = 1e-3;
            }
            println!("epoch {i}, loss {loss}");
            plot(&model, &param).unwrap();
            if loss.is_nan() {
                println!("loss is nan, something went wrong, exit");
                return;
            }
        }
    }
    println!("Done");
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
                .map(|x| (x, approchfun(x))),
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
