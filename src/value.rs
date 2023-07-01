use std::cell::RefCell;
use std::rc::Rc;

// `Value` is essentially just a wrapper around a floating point
// number that keeps track of the "operation graph" from which it
// was produced. The operation graph can then be traversed to determine
// the derivative (or "gradient") of each `Value` object that contributed
// to the final result.
//
// For example:
//
// let x = value(2.0);
// let y = value(3.0);
// let z = value(4.0);
//
// let out = x.mul(&y).add(&z);
// out.compute_gradients();
//
// println!(x.gradient()) -> 3.0
// println!(y.gradient()) -> 2.0
// println!(z.gradient()) -> 1.0
#[derive(Clone)]
pub struct Value {
    // In order to keep the borrow checker happy and make
    // the `Value` class more ergonomic to deal with, we wrap
    // all the underlying data in a reference counted mutable
    // cell. Rc<RefCell<T>> is more-or-less the equivalent of
    // a shared_ptr in C++ or a regular object in Python.
    // This is required since `Value` objects are self referential
    // and the same object can appear multiple times in the
    // operation graph.
    body: Rc<RefCell<ValueBody>>,
}

// Creates a new `Value` object from a constant.
pub fn value(x: f32) -> Value {
    Value::new(ValueBody {
        data: x,
        children: Vec::new(),
        gradient: 0.0,
        operation: None,
    })
}

impl Value {
    fn new(v: ValueBody) -> Value {
        Value {
            body: Rc::new(RefCell::new(v)),
        }
    }

    pub fn data(&self) -> f32 {
        self.body.borrow().data
    }

    pub fn gradient(&self) -> f32 {
        self.body.borrow().gradient
    }

    // Add the provided value.
    pub fn add(&self, v: &Value) -> Value {
        Value::new(ValueBody {
            data: self.data() + v.data(),
            children: vec![self.clone(), v.clone()],
            gradient: 0.0,
            operation: Some(Operation::Addition),
        })
    }

    // Subtract the provided value from this value.
    pub fn subtract(&self, v: &Value) -> Value {
        let negative = value(-1.0);
        let neg_v = v.mul(&negative);
        self.add(&neg_v)
    }

    // Raise this value to the provided power.
    pub fn pow(&self, p: &Value) -> Value {
        Value::new(ValueBody {
            data: self.data().powf(p.data()),
            children: vec![self.clone(), p.clone()],
            gradient: 0.0,
            operation: Some(Operation::Power),
        })
    }

    // Square this value.
    pub fn squared(&self) -> Value {
        let exponent = value(2.0);
        self.pow(&exponent)
    }

    // Multiply this value by the provided value.
    pub fn mul(&self, v: &Value) -> Value {
        Value::new(ValueBody {
            data: self.data() * v.data(),
            children: vec![self.clone(), v.clone()],
            gradient: 0.0,
            operation: Some(Operation::Multiplication),
        })
    }

    pub fn relu(&self) -> Value {
        Value::new(ValueBody {
            data: self.data().max(0.0),
            children: vec![self.clone()],
            gradient: 0.0,
            operation: Some(Operation::Relu),
        })
    }

    // Compute the gradients of all values in the operation graph
    // that contributed to this value.
    pub fn compute_gradients(&mut self) -> () {
        self.clear_gradients();
        // Since this value is the "root" of the graph it
        // always has a gradient of 1.
        self.body.borrow_mut().gradient = 1.0;
        self.compute_gradients_recursive();
    }

    // Assign the gradients to each child node and then recursively call this function
    // for each child. Note that all gradient computations here multiply the result
    // by self.gradient according to the power rule. In laymans terms, the power
    // rule says that you can chain derivatives together using multiplication.
    //
    // Conretely: If a biker can move 2x as fast as a walker, and a car is 4x faster than
    // a biker, then a car is 2 * 4 = 8 times faster than a walker.
    pub fn compute_gradients_recursive(&mut self) -> () {
        match self.body.borrow().operation {
            // Addition operations are gradient propagators.
            // E.g. in  x = y + z, y and z's
            // gradients w.r.t. to X are both just equal to
            // x's gradient w.r.t. x.
            Some(Operation::Addition) => {
                for child in &self.body.borrow().children {
                    child.body.borrow_mut().gradient += self.gradient();
                }
            }

            // Multiplication operations "swap" the gradients of the operands.
            // E.g. in x = y * z, y's gradient w.r.t. x is z, and z's gradient
            // w.r.t. x is y.
            Some(Operation::Multiplication) => {
                let lhs = &self.body.borrow().children[0];
                let rhs = &self.body.borrow().children[1];
                lhs.body.borrow_mut().gradient += rhs.data() * self.gradient();
                rhs.body.borrow_mut().gradient += lhs.data() * self.gradient();
            }

            // Power operations follow the classic derivative rule of:
            // dx/dy x = y^z => z * (y^(z-1))
            // Or more concretely: dy/dx y^3 = 2y^2
            Some(Operation::Power) => {
                let base = &self.body.borrow().children[0];
                let exponent = &self.body.borrow().children[1];
                base.body.borrow_mut().gradient +=
                    exponent.data() * (base.data().powf(exponent.data() - 1.0)) * self.gradient();
            }

            // The Relu function derivative is: 1 if x > 0 else 0
            Some(Operation::Relu) => {
                let base = &self.body.borrow().children[0];
                base.body.borrow_mut().gradient += if base.data() > 0.0 {
                    1.0 * self.gradient()
                } else {
                    0.0
                };
            }

            None => (),
        }
        for child in &mut self.body.borrow_mut().children {
            child.compute_gradients_recursive()
        }
    }

    // Move this value in the direction of the gradient proporitional to the provided
    // `learning_rate`.
    pub fn learn(&self, learning_rate: f32) {
        self.body.borrow_mut().data -= self.gradient() * learning_rate;
    }

    // Zero out all gradients in the operation graph.
    fn clear_gradients(&mut self) {
        self.body.borrow_mut().gradient = 0.0;
        for child in &mut self.body.borrow_mut().children {
            child.clear_gradients();
        }
    }

    // Dump the operation graph, just for internal debugging purposes.
    pub fn dump(&self) -> () {
        println!("data = {}, gradient = {}", self.data(), self.gradient());
        for child in &self.body.borrow().children {
            child.dump();
        }
    }
}

enum Operation {
    Addition,
    Multiplication,
    Power,
    Relu,
}

struct ValueBody {
    data: f32,
    children: Vec<Value>,
    gradient: f32,
    operation: Option<Operation>,
}
