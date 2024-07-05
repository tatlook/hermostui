
use std::{fs, io::Write};

use hermostui::{
    learning::Sequence,
    linealg::{Shape, Tensor, Vector},
    modules::{CrossEntropyLoss, Linear, ReLU, Sigmoid, Translation},
};
use mnist::MnistBuilder;

use rand::Rng;

const MNIST_IMAGE_SIZE: usize = 28 * 28;
const TRAINING_SET_LENGTH: u32 = 6000;
fn main() {
    let lr = 1e-2;

    let mnist::NormalizedMnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(TRAINING_SET_LENGTH)
        .validation_set_length(0)
        .test_set_length(0)
        .finalize()
        .normalize();
    let mut seq = Sequence::new(
        vec![
            Box::new(Linear),
            Box::new(Translation),
            Box::new(ReLU),
            Box::new(Linear),
            Box::new(Translation),
            Box::new(Sigmoid),
        ],
        vec![
            Tensor::rand(Shape::M(30, MNIST_IMAGE_SIZE)),
            Tensor::rand(Shape::V(30)),
            Tensor::N,
            Tensor::rand(Shape::M(10, 30)),
            Tensor::rand(Shape::V(10)),
            Tensor::N,
        ],
        Box::new(CrossEntropyLoss),
    );
    if let Ok(data) = fs::read("data/mnist.pkl") {
        println!("Resuming training from data/mnist.pkl");
        let params: Vec<Tensor> = serde_pickle::from_slice(&data, Default::default()).unwrap();
        seq.set_params(params);
    } else {
        println!("Training from scratch");
    }
    for i in 0..10000 {
        seq.zero_grad();
        for _ in 0..5 {
            let j = rand::thread_rng().gen_range(0..TRAINING_SET_LENGTH) as usize;
            let mut target = [0.0; 10];
            target[trn_lbl[j] as usize] = 1.0;

            let j = j * MNIST_IMAGE_SIZE;
            let input = &trn_img[j..(j + MNIST_IMAGE_SIZE)];

            seq.forward(Vector(Vec::from(input)));
            seq.backprop(Vector(Vec::from(target)));
        }
        seq.step(lr);
        
        if i % 200 == 0 {
            let loss = seq.get_loss();
            println!("epoch {i}, loss {loss}");
            
            println!("Check how well this model is in data/plot.bmp");
            draw(&seq, &trn_img, &trn_lbl);

            println!("Saving params");
            let binary = serde_pickle::to_vec(&seq.get_params(), Default::default()).unwrap();
            if let Ok(mut file) = std::fs::File::create("data/mnist.pkl") {
                file.write_all(&binary).unwrap();
            } else {
                println!("Failed to save params");
            }
        }
    }
}

fn model_output(seq: &Sequence, image: &[f32]) -> usize {
    let output = seq.evaluate(Vector(Vec::from(image)));

    let mut max = 0.0;
    let mut max_i = 0;
    for i in 0..10 {
        if output.0[i] > max {
            max = output.0[i];
            max_i = i;
        }
    }
    max_i
}

fn draw(seq: &Sequence, trn_img: &Vec<f32>, trn_lbl: &Vec<u8>) {
    use plotters::prelude::*;
    let root = BitMapBackend::new("data/plot.bmp", (300, 300)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)
        .unwrap();

    let mut accurecy = 0;
    for i in 0..36 {
        let offset_i = (i % 6, i / 6);
        let offset = (offset_i.0 as f64 / 6., offset_i.1 as f64 / 6.);

        let j: usize = rand::thread_rng().gen_range(0..TRAINING_SET_LENGTH) as usize;
        let img = &trn_img[j * MNIST_IMAGE_SIZE..j * MNIST_IMAGE_SIZE + MNIST_IMAGE_SIZE];

        scatter_ctx
            .draw_series(img.iter().enumerate().map(|(i, m)| {
                let x = (i % 28) as f64 / 300. + offset.0;
                let y = (i / 28) as f64 / 300. + offset.1;
                let y = 0.97 - y;
                let color = RGBColor((m * 255.0) as u8, 0, 0);
                Rectangle::new([(x, y), (x + 0.001, y + 0.001)], &color)
            }))
            .unwrap();

        let dot_and_label = |x: i32, y: i32, label: String| {
            return EmptyElement::at((x, y))
                + Text::new(label, (10, 0), ("sans-serif", 15.0).into_font());
        };
        let model_says = model_output(seq, img) as u8;
        let actual = trn_lbl[j];
        root.draw(&dot_and_label(
            offset_i.0 as i32 * 60,
            offset_i.1 as i32 * 55,
            format!("{}-{}", actual, model_says),
        ))
        .unwrap();

        if actual == model_says {
            accurecy += 1;
        }
    }

    root.present().unwrap();

    println!("Accurecy: {accurecy}/36, {}%", accurecy as f32 / 36.0 * 100.0);
}
