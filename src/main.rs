fn main() {
    let inputs: Vec<f32> = [1.0, 2.0].to_vec();
    let output: Vec<f32> = [300.0, 500.0].to_vec();

    let alpha: f32 = 0.01;
    let iterations = 10000;

    let w_init: f32 = 0.0;
    let b_init: f32 = 0.0;

    let (w_final, b_final, J_hist, p_hist) = gradient_descent(inputs, output, w_init, b_init, alpha,
                                                        iterations);

    println!("(w, b) found by gradient descent: ({}, {})", &w_final, &b_final);
}

fn compute_cost(sample_inputs: &Vec<f32>, sample_outputs: &Vec<f32>, w: &f32, b: &f32) -> f32 {
    let m = sample_inputs.len();
    let mut cost: f32 = 0 as f32;
    for i in 0..m {
        let current_sample = w * sample_inputs[i] + b - sample_outputs[i];
        cost += f32::powf(current_sample, 2.0);
    }

    let first: f32 = (1 / (2 * m)) as f32;
    first * cost
}

fn compute_gradient(sample_inputs: &Vec<f32>, sample_outputs: &Vec<f32>, w: &f32, b: &f32) -> (f32, f32) {
    let m = sample_inputs.len();
    let mut dj_dw: f32 = 0.0;
    let mut dj_db: f32 = 0.0;

    for i in 0..m {
        let f_wb = w * sample_inputs[i] + b;
        let dj_dw_i = (f_wb - sample_outputs[i]) * sample_inputs[i];
        let dj_db_i = f_wb - sample_outputs[i];

        dj_dw += dj_dw_i;
        dj_db += dj_db_i;
    }

    let m_clone = m.clone() as f32;

    (dj_dw / m_clone, dj_db / m_clone)
}

fn gradient_descent(sample_inputs: Vec<f32>,
                    sample_outputs: Vec<f32>,
                    w_in: f32,
                    b_in: f32,
                    alpha: f32,
                    num_iters: u32) -> (f32, f32, Vec<f32>, Vec<(f32, f32)>) {

    let mut j_history = Vec::<f32>::new();
    let mut p_history = Vec::<(f32, f32)>::new();
    let mut b = b_in.clone();
    let mut w = w_in.clone();

    for i in 0..num_iters {
        let (dj_dw, dj_db) = compute_gradient(&sample_inputs, &sample_outputs, &w, &b);

        b = b - alpha * dj_db;
        w = w - alpha * dj_dw;

        if i < 100000 {
            j_history.push(compute_cost(&sample_outputs, &sample_outputs, &w, &b));
            p_history.push((w, b));
        }

        if i.clone() as f32 % f32::ceil((num_iters / 20) as f32) == 0.0 {
            println!("Iteration {}: Cost {} dj_dw: {}, dj_db: {} w: {}, b:{}",
                                   &i, &j_history.last().unwrap(), &dj_dw, &dj_db, &w, &b);
        }
    }

    (w, b, j_history, p_history)
}