// fn loss(x: f32) -> f32 {
//     x.powi(2) + x + 1.
// }

// /// Better spices will take over worse spices
// fn randomchoice(spices: Vec<f32>, scores: Vec<f32>) -> Vec<f32> {
//     let mut rng = rand::thread_rng();
//     let all_scores = scores.iter().sum::<f32>();
//     for (spice, score) in zip(spices, scores) {
//         unimplemented!()
//     }
//     panic!()
// }

// fn main() {
//     let x = 500.0;
//     let mut spices = [x; 10];
//     for _ in 0..100000 {
//         for j in &mut spices {
//             *j += rand::thread_rng().gen_range(-0.1..0.1);
//         }
//         let mut min = f32::INFINITY;
//         let mut min_spice = 0.0;
//         for spice in spices {
//             let score = loss(spice);
//             if score < min {
//                 min = score;
//                 min_spice = spice;
//             }
//         }
//         spices = [min_spice; 10];
//     }
//     print!("{:?} ", spices);
// }

use std::fs;
use std::io::Write;
use std::iter::zip;

use hermostui::{
    learning::SGD,
    linealg::{Shape, Tensor, Vector},
    modules::{Linear, LossFunction, MSELoss, ReLU, Translation},
};

fn approchfun(x: f32) -> f32 {
    x / 20.
}

fn main() {
    let mut seq: SGD = SGD::new(
        vec![
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
            Tensor::rand(Shape::M(1, 16)),
            Tensor::rand(Shape::V(1)),
        ],
        Box::new(MSELoss),
    );
    if let Ok(data) = fs::read("data/sin_approch.pkl") {
        println!("Resuming training from data/sin_approch.pkl");
        let params: Vec<Tensor> = serde_pickle::from_slice(&data, Default::default()).unwrap();
        seq.set_params(params);
    } else {
        println!("Training from scratch");
    }
    let lr = 1e-1;
    for i in 0..10000 {
        let inputs: Vec<Vector> = (0..10)
            .map(|_| Vector(vec![rand::thread_rng().gen_range(-10.0..10.0)]))
            .collect();
        let targets: Vec<Vector> = inputs
            .iter()
            .map(|input| Vector(vec![approchfun(input.0[0])])).collect();
        let mut loss_sum_before = 0.0;
        for (input, target) in zip(&inputs, &targets) {
            let value = seq.evaluate(input.clone());
            let loss = MSELoss.loss(target, &value);
            loss_sum_before += loss;
        }
        let params_before = seq.get_params();
        let mut params_after = vec![];
        for param in &params_before {
            let rand = Tensor::rand(param.shape()) * lr;
            params_after.push(rand + param);
        }
        seq.set_params(params_after);

        let mut loss_sum_after = 0.0;
        for (input, target) in zip(&inputs, &targets) {
            let value = seq.evaluate(input.clone());
            let loss = MSELoss.loss(target, &value);
            loss_sum_after += loss;
        }
        if loss_sum_after > loss_sum_before {
            seq.set_params(params_before);
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
