use colored::*;
use rand::prelude::*;

fn compute_cost(data: &Vec<(u32, u32)>, w: f32) -> f32 {
    let mut acc_cost = 0f32;
    data.iter().for_each(|datum| {
        let y = datum.0 as f32 * w;
        let expected_y = datum.1 as f32;
        // cost is the squared delta of y and expected y
        let cost = (y - expected_y).powi(2);
        acc_cost += cost;
    });
    let avg_cost = acc_cost / data.len() as f32;
    avg_cost
}

fn main() {
    // We're trying to model y = x * w
    // x = training_data[i].0
    // y = training_data[i].1
    let training_data: Vec<(u32, u32)> =
        Vec::from([(0, 0), (1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]);

    // randomize weight w
    let mut rng = rand::thread_rng();
    let mut w = rng.gen::<f32>() * 100f32;

    // The goal is to reduce the average squared differences to zero.
    // A cost function of 0 means that our model predicts y with 100% accuracy.
    // In order to achieve this we need to nudge `w` in a direction that
    // drives down the cost function.
    let eps = 0.001f32;
    let learning_rate = 0.001f32;

    println!(
        "Randomized value for {}: {}",
        "w".red(),
        w.to_string().blue()
    );

    for _ in 0..10 {
        let saved_w = w;
        let cost = compute_cost(&training_data, w);
        let dcost = (compute_cost(&training_data, w + eps) - compute_cost(&training_data, w)) / eps;
        w -= dcost * learning_rate;
        println!(
            "Average Cost: {}, Dcost: {}, w: {}, w': {}",
            cost.to_string().blue(),
            dcost.to_string().blue(),
            saved_w.to_string().blue(),
            w.to_string().blue()
        );
    }
}
