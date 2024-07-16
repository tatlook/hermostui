Hermostui
=========

Hermostui is Tatlook's own small deep learning library. It is barelly functional.
I implemented a small liner algebra, backpropagation and of course neuron network. Non-gradient optimization is also supported.
Model's parameters can be serialized using `serde`, so it can be saved just like in PyTorch.

## Example
I also wrote some examples in `./examples/`, welcome to check. Below is a linear regression:
```rust
let mut model = Sequence::new(
    vec![ Box::new(Linear::new(1, 1)), Box::new(Translation::new(1)), ]
);
let mut param = Tensor::rand(model.param_shape());
let mut optim = SGD::new();
let lr = 1e-2;
for _ in 0..100 {
    let inputs = (0..50).map(|i| Vector(vec![i as f32 / 2.5 - 10.0]));
    let targets = inputs.clone().map(|x| Vector(vec![2.0 * x.0[0] + 1.0));
    optim.step(
        &mut model, &mut param, &MSELoss,
        inputs, targets, lr,
    );
}
```
This will let our model behave like function `2x+1`. You will notice there is no one affline
transform, but linear transform and translation sepretly. This is because in this way it is
easyer to implement, I think.
This example uses `SGD`optimizer, I also implemented greedy and genetic algorithm.

## Installization

Add this into your `Cargo.toml`:
```toml
[dependencies]
hermostui = { version = "0.1.0", git = "https://github.com/tatlook/hermostui.git"}
```

## License
MIT license, see `./LICENSE` for more.
