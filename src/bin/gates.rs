use colored::*;
use rand::prelude::*;

const ITERATIONS: usize = 1_000_001;

type TData = Vec<(u8, u8, u8)>;

fn sigmoid(x: f32) -> f32 {
    1f32 / (1f32 + (-1f32 * x).exp())
}

fn compute_cost(data: &TData, w1: f32, w2: f32, b: f32) -> f32 {
    let mut acc_cost = 0f32;
    data.iter().for_each(|datum| {
        let x1 = datum.0 as f32 * w1;
        let x2 = datum.1 as f32 * w2;
        let y = sigmoid(x1 + x2 + b);
        let expected_y = datum.2 as f32;
        let cost = (y - expected_y).powi(2);
        acc_cost += cost;
    });
    let avg_cost = acc_cost / data.len() as f32;
    avg_cost
}

fn main() {
    let training_data: TData = Vec::from([(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]);

    let mut rng = rand::thread_rng();
    let mut w1 = rng.gen::<f32>();
    let mut w2 = rng.gen::<f32>();
    let mut b = rng.gen::<f32>();
    let eps = 0.01f32;
    let learning_rate = 0.01f32;

    println!(
        "Randomized value for w1: {}, w2: {}, b: {}",
        w1.to_string().blue(),
        w2.to_string().blue(),
        b.to_string().blue()
    );
    println!(
        "Initial cost: {}",
        compute_cost(&training_data, w1, w2, b).to_string().blue()
    );
    println!("Initial run:");
    training_data.iter().for_each(|expr| {
        println!(
            "{} || {} -> {}",
            expr.0.to_string().blue(),
            expr.1.to_string().blue(),
            sigmoid(expr.0 as f32 * w1 + expr.1 as f32 * w2 + b)
                .to_string()
                .blue()
        );
    });

    println!("TRAINING...");
    for i in 1..ITERATIONS {
        let cost = compute_cost(&training_data, w1, w2, b);
        let dcost_w1 = (compute_cost(&training_data, w1 + eps, w2, b) - cost) / eps;
        let dcost_w2 = (compute_cost(&training_data, w1, w2 + eps, b) - cost) / eps;
        let dcost_b = (compute_cost(&training_data, w1, w2, b + eps) - cost) / eps;
        w1 -= dcost_w1 * learning_rate;
        w2 -= dcost_w2 * learning_rate;
        b -= dcost_b * learning_rate;
        if i % (ITERATIONS / 10) == 0 {
            println!(
                "[GEN {}] cost: {}, w1: {}, w2: {}, b: {}",
                i.to_string().green(),
                cost.to_string().red(),
                w1.to_string().blue(),
                w2.to_string().blue(),
                b.to_string().blue()
            );
        }
    }
    println!("Final run:");
    training_data.iter().for_each(|expr| {
        println!(
            "{} || {} -> {}",
            expr.0.to_string().blue(),
            expr.1.to_string().blue(),
            sigmoid(expr.0 as f32 * w1 + expr.1 as f32 * w2 + b)
                .to_string()
                .blue()
        );
    })
}
