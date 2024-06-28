use rand::Rng;


/// Neural network
struct NeuralNetwork {
    layers: Vec<Layer>,
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl NeuralNetwork {
    fn new(layer_sizes: Vec<i32>) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i + 1], layer_sizes[i]));
        }
        Self { layers }
    }

    fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        let mut outputs = inputs.to_vec();
        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }
        outputs
    }
}

impl Layer {
    fn new(neuron_count: i32, input_count: i32) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..neuron_count {
            let neuron = Neuron::new(vec![0.0; input_count as usize], 0.0);
            neurons.push(neuron);
        }
        Self { neurons }
    }

    fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        assert_eq!(inputs.len(), self.neurons[0].weights.len());
        let mut outputs = Vec::new();
        for neuron in &self.neurons {
            outputs.push(neuron.forward(inputs));
        }
        outputs
    }
}

struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

fn relu(x: f32) -> f32 {
    return x.max(0.0);
}

impl Neuron {
    fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self { weights, bias }
    }

    fn forward(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len(), "Invalid number of inputs");
        relu(dot(inputs, self.weights.as_slice()) + self.bias)
    }
}

trait Randomize {
    /// Randomize everything in this object
    fn randomize(&mut self);
}

impl Randomize for NeuralNetwork {
    fn randomize(&mut self) {
        for layer in &mut self.layers {
            layer.randomize();
        }
    }
}

impl Randomize for Layer {
    fn randomize(&mut self) {
        for neuron in &mut self.neurons {
            neuron.randomize();
        }
    }
}

impl Randomize for Neuron {
    fn randomize(&mut self) {
        for w in &mut self.weights {
            *w = rand::thread_rng().gen_range(-10.0 .. 10.0);
        }
        self.bias = rand::thread_rng().gen_range(-10.0 .. 10.0);
    }
}

fn backpropagation(nn: &NeuralNetwork) {

}

mod tests {
    #[test]
    fn test_sigmoid() {
        assert_eq!(super::relu(0.0), 0.5);
    }

    #[test]
    fn test_layer_forward() {
        let layer = super::Layer::new(3, 2);
        let inputs = vec![1.0, 1.0, 1.0];
        let outputs = layer.forward(&inputs);
        assert_eq!(outputs, [0.5, 0.5]);
    }

    #[test]
    fn test_neuron_forward() {
        let neuron = super::Neuron::new(vec![1.0, 2.0], 3.0);
        let inputs = vec![4.0, 5.0];
        let output = neuron.forward(&inputs);
        assert_eq!(output, 1.0);
    }

    #[test]
    fn test_neural_network() {
        let network = super::NeuralNetwork::new(vec![2, 3, 2]);
        let output = network.forward(&[0.1, 0.4]);
        assert_eq!(output, [0.5, 0.5]);
    }
}
