use std::fs;

use hermostui::{
    learning::{count_loss, SGD},
    linealg::{Tensor, Vector},
    modules::{Function, Linear, MSELoss, Sequence, Sigmoid, Translation, ELU},
};

fn approchfun(x: f32) -> f32 {
    static mut V: f32 = 1.0;
    unsafe {
        // V += 0.000003;
        return (x * V).sin() * 6.0 - 1.0;
    }
}

fn main() {
    let mut model = Sequence::new(vec![
        Box::new(Linear::new(1, 5)),
        Box::new(Translation::new(5)),
        Box::new(ELU { alpha: 1.0 }),
        Box::new(Linear::new(5, 15)),
        Box::new(Translation::new(15)),
        Box::new(Sigmoid),
        Box::new(Linear::new(15, 15)),
        Box::new(Translation::new(15)),
        Box::new(ELU { alpha: 1.0 }),
        Box::new(Linear::new(15, 1)),
        Box::new(Translation::new(1)),
    ]);
    let mut optim = SGD::new();
    let mut param: Tensor = if let Ok(data) = fs::read("data/sin_approch.pkl") {
        println!("Resuming training from data/sin_approch.pkl");
        serde_pickle::from_slice(&data, Default::default()).unwrap()
    } else {
        println!("Training from scratch");
        Tensor::rand(model.param_shape())
    };
    let loss_fn = MSELoss;
    let inputs = (0..50).map(|i| Vector(vec![i as f32 / 2.5 - 10.0]));
    let targets = inputs.clone().map(|x| Vector(vec![approchfun(x.0[0])]));
    let lr = 3e-2;
    for i in 0..10000 {
        optim.eat_batch(&mut model, &param, &loss_fn, inputs.clone(), targets.clone());
        optim.step(&mut param, lr);

        if i % 100 == 0 {
            let loss = count_loss(&model, &param, &loss_fn, inputs.clone(), targets.clone());
            println!("epoch {i}, loss {loss}");
            plot(&model, &param).unwrap();
            if loss.is_nan() {
                println!("loss is nan, something went wrong, exit");
                return;
            }
        }
    }
    println!("Done");
    // println!("Saving params");
    // let binary = serde_pickle::to_vec(&seq.get_param(), Default::default()).unwrap();
    // if let Ok(mut file) = std::fs::File::create("data/sin_approch.pkl") {
    //     file.write_all(&binary).unwrap();
    // } else {
    //     println!("Failed to save params");
    // }
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
