mod value;
use value::value;

mod neural_net;
use neural_net::NeuralNet;

fn main() {
    let net = NeuralNet::new(vec![2, 1]);
    for i in (0..1000) {
        let inputs = vec![
            vec![value(5.0), value(5.0)],
            vec![value(4.0), value(3.0)],
            vec![value(10.0), value(3.0)],
            vec![value(-15.0), value(3.0)],
            vec![value(-5.0), value(3.0)],
        ];
        let expected_outputs = vec![
            value(10.0),
            value(7.0),
            value(13.0),
            value(-12.0),
            value(-2.0),
        ];
        let square_error = inputs
            .iter()
            .map(|input| net.forward(input.clone()))
            .zip(expected_outputs)
            .map(|(predicted, actual)| {
                predicted[0].subtract(&actual).squared()
            }).reduce(|x, y| x.add(&y))
            .unwrap();
        let inverse_size = value(1.0 / (inputs.len() as f32));
        let mut mean_square_error = square_error.mul(&inverse_size);
        println!("loss={}", mean_square_error.data());
        mean_square_error.compute_gradients();
        mean_square_error.learn(0.0001);
    }

    net.dump();
    println!("9 + 4 = {}", net.forward(vec![value(9.0), value(4.0)])[0].data());
}
