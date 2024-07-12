use std::fs;
use std::io::Write;

use hermostui::{
    learning::SGD,
    linealg::{Shape, Tensor, Vector},
    modules::{Linear, MSELoss, Sequence, Translation, ELU},
};
fn approchfun(x: f32) -> f32 {
    static mut V: f32 = 1.0;
    unsafe {
        V += 0.000003;
        return (x * V).sin() * 6.0 - 1.0;
    }
}

fn main() {
    let mut optim: SGD = SGD::new(Box::new(Sequence::new(vec![
            Box::new(Linear),
            Box::new(Translation),
            Box::new(ELU { alpha: 1.0 }),
            Box::new(Linear),
            Box::new(Translation),
            Box::new(ELU { alpha: 1.0 }),
            Box::new(Linear),
            Box::new(Translation),
            Box::new(ELU { alpha: 1.0 }),
            Box::new(Linear),
            Box::new(Translation),
            Box::new(ELU { alpha: 1.0 }),
            Box::new(Linear),
            Box::new(Translation),
        ])),
        Tensor::L(vec![
            Tensor::rand(Shape::M(5, 1)),
            Tensor::rand(Shape::V(5)),
            Tensor::N,
            Tensor::rand(Shape::M(15, 5)),
            Tensor::rand(Shape::V(15)),
            Tensor::N,
            Tensor::rand(Shape::M(5, 15)),
            Tensor::rand(Shape::V(5)),
            Tensor::N,
            Tensor::rand(Shape::M(15, 5)),
            Tensor::rand(Shape::V(15)),
            Tensor::N,
            Tensor::rand(Shape::M(1, 15)),
            Tensor::rand(Shape::V(1)),
        ]),
        Box::new(MSELoss),
    );
    if let Ok(data) = fs::read("data/sin_approch.pkl") {
        println!("Resuming training from data/sin_approch.pkl");
        let param: Tensor = serde_pickle::from_slice(&data, Default::default()).unwrap();
        optim.set_param(param);
    } else {
        println!("Training from scratch");
    }
    let lr = 1e-3;
    for i in 0..10000 {
        optim.zero_grad();
        for _ in 0..5 {
            let x = rand::thread_rng().gen_range(-10.0..10.0);
            let y = approchfun(x);
            optim.forward(Vector(vec![x]));
            optim.backprop(Vector(vec![y]), Vector(vec![x]));
        }
        optim.step(lr);

        if i % 100 == 0 {
            let loss = optim.get_loss();
            println!("epoch {i}, loss {loss}");
            plot(&mut optim).unwrap();
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
use rand::Rng;
fn plot(seq: &mut SGD) -> Result<(), Box<dyn std::error::Error>> {
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
