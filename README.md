Hermostui
=========

Hermostui is Tatlook's own small deep learning library. It is barelly functional.
I implemented a small liner algebra, backpropagation and of course neuron network.
Model's parameters can be serialized using `serde`, so it can be saved just like in PyTorch.

## Example
I also wrote some examples in `./examples/`, welcome to check. Below is a linear regression:
```rust
let mut seq = Sequence::new(
    vec![ Box::new(Linear), Box::new(Translation), ],
    vec![ Tensor::rand(Shape::M(1, 1)), Tensor::rand(Shape::V(1)), ],
    Box::new(MSELoss),
);
let lr = 1e-2;
for _ in 0..100 {
    seq.zero_grad();
    for _ in 0..10 {
        let x = rand::thread_rng().gen_range(-10.0..10.0);
        let y = x * 2.0 + 1.0;
        seq.forward(Tensor::S(x));
        seq.backprop(Tensor::S(y));
    }
    seq.step(lr);
}
```
This will let our model behave like function `2x+1`. You will notice there is no one affline
transform, but linear transform and translation sepretly. This is because in this way it is
easyer to implement, I think.
I call such a struct `Sequence`, because I don't know what name I should give it.
Also a pretty bad idea to give functions and parametres sepretly, since this every time I
modify functions I have to modify parametries.

## Installization

Add this into your `Cargo.toml`:
```toml
[dependencies]
hermostui = { version = "0.1.0", git = "https://github.com/tatlook/hermostui.git"}
```

## License
MIT license, see `./LICENSE` for more.
