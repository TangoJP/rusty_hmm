use rusty_hmm::{hmm, log_hmm, generative, utility};
use ndarray::{Axis, arr2};


#[test]
// #[ignore]
fn test_log_forward_backward() {
    let len_seq = 10000;
    let init_dist = vec![0.65, 0.35];
    let trans_mat = arr2(&[
        [0.8, 0.2],
        [0.2, 0.8]
    ]);
    let emit_mat = arr2(&[
        [0.99, 0.01],
        [0.01, 0.99]
    ]);
    
    let state_seq = generative::generate_state_sequence(&init_dist, &trans_mat, len_seq);
    let obs_seq = generative::generate_observation_sequence(&state_seq, &emit_mat);
    
    let mut obs = Vec::<u8>::new();
    for o in obs_seq.iter() {
        obs.push(*o as u8);
    }

    // run forward_backward with the 'actual' trans_mat and emit_mat
    let iteration = 500;
    let mut init_dist_hat = vec![0.5, 0.5];
    let mut trans_mat_hat = arr2(&[
        [0.7, 0.3],
        [0.3, 0.7]
    ]);
    let mut emit_mat_hat = arr2(&[
        [0.7, 0.3],
        [0.3, 0.7]
    ]);
    let (init_hat, a_hat, b_hat) = log_hmm::log_compute_forward_backward(
        &obs, 
        &mut init_dist_hat, 
        &mut trans_mat_hat, 
        &mut emit_mat_hat, 
        iteration);
    
    println!("Actual init_mat\n{:?}", init_hat);
    println!("Estimated init_mat\n{:?}", init_dist_hat);

    println!("Actual trans_mat\n{:?}", trans_mat);
    println!("Estimated trans_mat\n{:?}", a_hat);

    println!("Actual emit_mat\n{:?}", emit_mat);
    println!("Estimated emit_mat\n{:?}", b_hat);
    
}


#[test]
#[ignore]
fn test_forward_backward() {
    let len_seq = 500;
    let init_dist = vec![0.34, 0.33, 0.33];
    let trans_mat = arr2(&[
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6]
    ]);
    let emit_mat = arr2(&[
        [0.25, 0.25, 0.25, 0.25],
        [0.49, 0.01, 0.49, 0.01],
        [0.01, 0.49, 0.01, 0.49]
    ]);
    
    let state_seq = generative::generate_state_sequence(&init_dist, &trans_mat, len_seq);
    let obs_seq = generative::generate_observation_sequence(&state_seq, &emit_mat);
    
    let mut obs = Vec::<u8>::new();
    for o in obs_seq.iter() {
        obs.push(*o as u8);
    }

    // run forward_backward with the 'actual' trans_mat and emit_mat
    let iteration = 100;
    let mut trans_mat_hat = arr2(&[
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]
    ]);
    let mut emit_mat_hat = arr2(&[
        [0.25, 0.25, 0.25, 0.25],
        [0.4, 0.1, 0.4, 0.1],
        [0.1, 0.4, 0.1, 0.4]
    ]);
    let (a_hat, b_hat) = hmm::forward_backward(
        &obs, 
        &init_dist, 
        &mut trans_mat_hat, 
        &mut emit_mat_hat, 
        iteration);
    
    println!("Actual trans_mat\n{:?}", trans_mat);
    println!("Estimated trans_mat\n{:?}", a_hat);

    println!("Actual emit_mat\n{:?}", emit_mat);
    println!("Estimated emit_mat\n{:?}", b_hat);
    
}


#[test]
#[ignore]
// Check if forward and backward probabilities would match
fn test_regular_vs_log_probs() {
    // Create a mock sequence
    let len_seq = 100;
    let init_dist = vec![0.4, 0.3, 0.3];
    let trans_mat = arr2(&[
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]
    ]);
    let emit_mat = arr2(&[
        [0.25, 0.25, 0.25, 0.25],
        [0.49, 0.01, 0.49, 0.01],
        [0.01, 0.49, 0.01, 0.49]
    ]);
    
    let state_seq = generative::generate_state_sequence(&init_dist, &trans_mat, len_seq);
    let obs_seq = generative::generate_observation_sequence(&state_seq, &emit_mat);
    
    let obs = obs_seq.iter().map(|x| *x as u8).collect();
    
    // calculate regular forward & backforward probabilities
    let forward_prob = hmm::get_forward_prob(
        &hmm::forward(&obs, &init_dist, &trans_mat, &emit_mat)
    );
    let backward_prob = hmm::get_backward_prob(
        &hmm::backward(&obs, &trans_mat, &emit_mat),
        &obs, &init_dist, &emit_mat
    );

    let log_forward_prob = log_hmm::log_compute_forward_prob(
        &log_hmm::log_compute_forward_matrix(&obs, &init_dist, &trans_mat, &emit_mat)
    );

    println!("Computing log_backward_prob");
    let log_backward_prob = log_hmm::loc_compute_backward_prob(
        &log_hmm::log_compute_backward_matrix(&obs, &trans_mat, &emit_mat),
        &obs, &init_dist, &emit_mat
    );


    println!("Forward  probability (ln(regular) vs ln_prob): {:?} vs {:?}", forward_prob.ln(), log_forward_prob);
    println!("Backward probability (ln(regular) vs ln_prob): {:?} vs {:?}", backward_prob.ln(), log_backward_prob);

}


#[test]
#[ignore]
// Check if forward and backward probabilities would match
fn test_forward_and_backward_probs() {
    // Create a mock sequence
    let len_seq = 100;
    let init_dist = vec![0.4, 0.3, 0.3];
    let trans_mat = arr2(&[
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]
    ]);
    let emit_mat = arr2(&[
        [0.25, 0.25, 0.25, 0.25],
        [0.49, 0.01, 0.49, 0.01],
        [0.01, 0.49, 0.01, 0.49]
    ]);
    
    let state_seq = generative::generate_state_sequence(&init_dist, &trans_mat, len_seq);
    let obs_seq = generative::generate_observation_sequence(&state_seq, &emit_mat);
    
    let obs = obs_seq.iter().map(|x| *x as u8).collect();

    let forward_prob = hmm::get_forward_prob(
        &hmm::forward(&obs, &init_dist, &trans_mat, &emit_mat)
    );
    let backward_prob = hmm::get_backward_prob(
        &hmm::backward(&obs, &trans_mat, &emit_mat),
        &obs, &init_dist, &emit_mat
    );

    println!("Forward  probability = {:?}", forward_prob.log10());
    println!("Backward probability = {:?}", backward_prob.log10());

}

#[test]
#[ignore]
fn test_forward() {
    // thinking coin flip with a normal (state 0) and fixed coin (state 1)
    // assume observation 1 = head, 0 = tail
    let obs = vec![0u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    let mut init_dist = Vec::<f64>::new();
    init_dist.push(0.65);
    init_dist.push(0.35);

    let trans_mat = arr2(&[
        [0.6, 0.4],
        [0.4, 0.6]
    ]);
    let emit_mat = arr2(&[
        [0.5, 0.5],
        [0.8, 0.2]
    ]);

    let forward_mat = hmm::forward(
         &obs, 
         &init_dist, 
         &trans_mat, 
         &emit_mat);
    
    let forward_prob = forward_mat.sum_axis(Axis(0))[obs.len() - 1];

    println!("Forward Matrix with shape{:?}\n{:?}", forward_mat.shape(), forward_mat);
    println!("Forward probability = {:?}", forward_prob);

}

#[test]
#[ignore]
fn test_backward() {
    // thinking coin flip with a normal (state 0) and fixed coin (state 1)
    // assume observation 1 = head, 0 = tail
    let obs = vec![0u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 1u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    let mut init_dist = Vec::<f64>::new();
    init_dist.push(0.65);
    init_dist.push(0.35);

    let trans_mat = arr2(&[
        [0.6, 0.4],
        [0.4, 0.6]
    ]);
    let emit_mat = arr2(&[
        [0.5, 0.5],
        [0.8, 0.2]
    ]);

    let backward_mat = hmm::backward(
         &obs, 
         &trans_mat, 
         &emit_mat);
    
    let backward_prob = hmm::get_backward_prob(
    &backward_mat, 
    &obs,
    &init_dist,
    &emit_mat
    );

    println!("Backward Matrix with shape{:?}\n{:?}", backward_mat.shape(), backward_mat);
    println!("Backward probability = {:?}", backward_prob);

}

#[test]
#[ignore]
fn test_viterbi() {
    // thinking coin flip with a normal (state 0) and fixed coin (state 1)
    // assume observation 1 = head, 0 = tail
    let obs = vec![0u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 1u8, 1u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    let mut init_dist = Vec::<f64>::new();
    init_dist.push(0.5);
    init_dist.push(0.5);

    let trans_mat = arr2(&[
        [0.55, 0.45],
        [0.45, 0.55]
    ]);
    let emit_mat = arr2(&[
        [0.5, 0.5],
        [0.65, 0.35]
    ]);

    let (v, b) = hmm::viterbi(
         &obs, 
         &init_dist, 
         &trans_mat, 
         &emit_mat);
    
    let mut best_path_prob = 0.0;
    let mut best_bp = 0;
    for (i, row) in v.axis_iter(Axis(0)).enumerate() {
        let prob_temp = row[obs.len() - 1];
        if  prob_temp > best_path_prob {
            best_path_prob = prob_temp;
            best_bp = i;
        }
    }

    let bestpath = hmm::traceback_viterbi(&b, best_bp);

    println!("Viterbi Matrix with shape{:?}\n{:?}", v.shape(), v);
    println!("Backtrace Matrix with shape{:?}\n{:?}", b.shape(), b);

    println!("Viterbi probability = {:?}", best_path_prob);
    println!("Best Path = {:?}", bestpath);


}


