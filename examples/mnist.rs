use std::{fs, io::Write};

use hermostui::{
    learning::{count_loss, Optimizer, SGD},
    linealg::{Tensor, Vector},
    modules::{CrossEntropyLoss, Function, Linear, ReLU, Sequence, Softmax, Translation},
};
use mnist::MnistBuilder;

use rand::{thread_rng, Rng};

const MNIST_IMAGE_SIZE: usize = 28 * 28;
const TRAINING_SET_LENGTH: u32 = 60000;
const TEST_SET_LENGTH: u32 = 10000;

fn main() {
    let mnist::NormalizedMnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(TRAINING_SET_LENGTH)
        .validation_set_length(0)
        .test_set_length(TEST_SET_LENGTH)
        .finalize()
        .normalize();
    let mut model = Sequence::new(vec![
        Box::new(Linear::new(MNIST_IMAGE_SIZE, 30)),
        Box::new(Translation::new(30)),
        Box::new(ReLU),
        Box::new(Linear::new(30, 10)),
        Box::new(Translation::new(10)),
        Box::new(Softmax::new()),
    ]);
    let mut param: Tensor = if let Ok(data) = fs::read("data/mnist.pkl") {
        println!("Resuming training from data/mnist.pkl");
        serde_pickle::from_slice(&data, Default::default()).unwrap()
    } else {
        println!("Training from scratch");
        Tensor::rand(model.param_shape())
    };
    let mut optim = SGD::new();
    let lr = 1e-1;
    let loss_fn = CrossEntropyLoss;
    let batch_size = 30;
    for i in 0..10000 {
        let j = i % TRAINING_SET_LENGTH as usize;
        let indexes = j..j + batch_size;
        let inputs = indexes.clone().map(|j| {
            Vector(Vec::from(
                &trn_img[j * MNIST_IMAGE_SIZE..(j + 1) * MNIST_IMAGE_SIZE],
            ))
        });
        let targets = indexes.map(|j| {
            let mut target = [0.0; 10];
            target[trn_lbl[j] as usize] = 1.0;
            Vector(Vec::from(target))
        });
        optim.step(
            &mut model,
            &mut param,
            &loss_fn,
            inputs.clone(),
            targets.clone(),
            lr,
        );

        if i % 100 == 0 {
            let loss = count_loss(&mut model, &param, &loss_fn, inputs, targets);
            println!("epoch {i}, loss {loss}");

            println!(
                "Train accuracy: {}/1000",
                accurecy1000(&model, &param, &trn_img, &trn_lbl)
            );
            println!(
                "Test accuracy: {}/1000",
                accurecy1000(&model, &param, &tst_img, &tst_lbl)
            );
            println!("Check how well this model is in data/plot.bmp");
            draw(&model, &param, &tst_img, &tst_lbl);

            println!("Saving params");
            let binary = serde_pickle::to_vec(&param, Default::default()).unwrap();
            if let Ok(mut file) = std::fs::File::create("data/mnist.pkl") {
                file.write_all(&binary).unwrap();
            } else {
                println!("Failed to save params");
            }
        }
    }
}

fn model_output(model: &dyn Function, param: &Tensor, image: &[f32]) -> usize {
    let output = model.evaluate(param, &Vector(Vec::from(image)));

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

fn accurecy1000(
    model: &dyn Function,
    param: &Tensor,
    tst_img: &Vec<f32>,
    tst_lbl: &Vec<u8>,
) -> usize {
    let mut accurecy = 0;
    for i in 0..1000 {
        let i = i as usize;
        let img = &tst_img[i * MNIST_IMAGE_SIZE..i * MNIST_IMAGE_SIZE + MNIST_IMAGE_SIZE];
        let model_says = model_output(model, param, img) as u8;
        let actual = tst_lbl[i];
        if actual == model_says {
            accurecy += 1;
        }
    }
    accurecy
}

fn draw(model: &dyn Function, param: &Tensor, tst_img: &Vec<f32>, tst_lbl: &Vec<u8>) {
    use plotters::prelude::*;
    let root = BitMapBackend::new("data/plot.bmp", (600, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut scatter_ctx = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)
        .unwrap();

    for i in 0..100 {
        let j = thread_rng().gen_range(0..TEST_SET_LENGTH) as usize;
        let img = &tst_img[j * MNIST_IMAGE_SIZE..j * MNIST_IMAGE_SIZE + MNIST_IMAGE_SIZE];
        let model_says = model_output(model, param, img) as u8;
        let actual = tst_lbl[j];

        let offset_i = (i % 10, i / 10);
        let offset = (offset_i.0 as f64 / 10., offset_i.1 as f64 / 10.);

        scatter_ctx
            .draw_series(img.iter().enumerate().map(|(i, m)| {
                let x = (i % 28) as f64 / 600. + offset.0;
                let y = (i / 28) as f64 / 600. + offset.1;
                let y = 0.97 - y;
                let color = RGBColor((m * 255.0) as u8, 0, 0);
                Rectangle::new([(x, y), (x + 0.001, y + 0.001)], &color)
            }))
            .unwrap();

        let dot_and_label = |x: i32, y: i32, label: String| {
            return EmptyElement::at((x, y))
                + Text::new(label, (10, 0), ("sans-serif", 15.0).into_font());
        };

        root.draw(&dot_and_label(
            offset_i.0 as i32 * 60,
            offset_i.1 as i32 * 55,
            format!("{}-{}", actual, model_says),
        ))
        .unwrap();
    }

    root.present().unwrap();
}
