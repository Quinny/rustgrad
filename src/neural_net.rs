// A basic neural network package which leverages the `Value` class
// as it's base element.

use std::iter::zip;
use crate::value::{value, Value};
use rand::{self, Rng};

// A single neuron which multiplies each input feature
// against each weight and adds the bias.
struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    fn new(parameters: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        Neuron {
            weights: (0..parameters)
                .map(|_x| value(rng.gen_range(-1.0..1.0)))
                .collect(),
            bias: value(rng.gen_range(-1.0..1.0)),
        }
    }

    fn forward(&self, inputs: &Vec<Value>) -> Value {
        zip(&self.weights, inputs)
            .map(|(w, i)| w.mul(&i))
            .reduce(|x, y| x.add(&y))
            .unwrap()
            .add(&self.bias)
    }
}

// A layer of neurons. 
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Layer {
        Layer {
            neurons: (0..output_size).map(|_x| Neuron::new(input_size)).collect(),
        }
    }

    fn forward(&self, inputs: &Vec<Value>) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }
}

// A collection of layers.
pub struct NeuralNet {
    layers: Vec<Layer>,
}

impl NeuralNet {
    pub fn new(layer_sizes: Vec<usize>) -> NeuralNet {
        NeuralNet {
            layers: (0..layer_sizes.len() - 1)
                .map(|layer_index| {
                    Layer::new(layer_sizes[layer_index], layer_sizes[layer_index + 1])
                })
                .collect(),
        }
    }

    pub fn forward(&self, inputs: Vec<Value>) -> Vec<Value> {
        let mut output = inputs;
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn dump(&self) {
        for layer in &self.layers {
            println!("layer");
            for neuron in &layer.neurons {
                let ws: Vec<f32> = neuron.weights.clone().into_iter().map(|w| w.data()).collect();
                println!("w={:?}, b={}", ws, neuron.bias.data());
            }
        }
    }
}
