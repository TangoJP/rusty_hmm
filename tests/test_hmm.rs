use rusty_hmm::hmm::HMM;
use rusty_hmm::utility::{eexpo};//check_prob_matrix_sums_to_one, check_prob_vector_sums_to_one};
use ndarray::{Axis, arr2};
use float_cmp::approx_eq;

mod common;

#[test]
// #[ignore]
fn test_hmm1() {
    let (model, 
        (init_dist_hat, 
        trans_mat_hat, 
        emit_mat_hat)
    ) = common::case1();
    
    let obs = model.observations;
    let iteration = 100;

    let mut hmm = HMM::new(
        obs,
        init_dist_hat,
        trans_mat_hat,
        emit_mat_hat,
        iteration
    );

    hmm.train();
    
    println!("==================== Testing HMM ====================");
    println!("Actual init_mat\n{:?}", model.init_dist);
    println!("Estimated init_mat\n{:?}", hmm.init_dist);

    println!("Actual trans_mat\n{:.3e}", model.trans_mat);
    println!("Estimated trans_mat\n{:.3e}", hmm.trans_mat);

    println!("Actual emit_mat\n{:.3e}", model.emit_mat);
    println!("Estimated emit_mat\n{:.3e}", hmm.emit_mat);
    
    println!("trans_mat rows add up to 1.0?: {:.4e}", hmm.trans_mat.sum_axis(Axis(1)));
    println!("emit_mat rows add up to 1.0?: {:.4e}", hmm.emit_mat.sum_axis(Axis(1)));

    println!("Actual state sequence (first 50): {:?}", &model.state_sequence[0..50]);
    println!("Estimated state sequence (first 50): {:?}", &hmm.best_state_sequence[0..50]);

}


#[test]
#[ignore]
fn test_hmm2() {
    let (model, 
        (init_dist_hat, 
        trans_mat_hat, 
        emit_mat_hat)
    ) = common::case2();
    
    let obs = model.observations;
    let iteration = 100;

    let mut hmm = HMM::new(
        obs,
        init_dist_hat,
        trans_mat_hat,
        emit_mat_hat,
        iteration
    );

    hmm.train();
    
    println!("==================== Testing HMM ====================");
    println!("Actual init_mat\n{:?}", model.init_dist);
    println!("Estimated init_mat\n{:?}", hmm.init_dist);

    println!("Actual trans_mat\n{:.3e}", model.trans_mat);
    println!("Estimated trans_mat\n{:.3e}", hmm.trans_mat);

    println!("Actual emit_mat\n{:.3e}", model.emit_mat);
    println!("Estimated emit_mat\n{:.3e}", hmm.emit_mat);
    
    println!("trans_mat rows add up to 1.0?: {:.4e}", hmm.trans_mat.sum_axis(Axis(1)));
    println!("emit_mat rows add up to 1.0?: {:.4e}", hmm.emit_mat.sum_axis(Axis(1)));
}


#[test]
#[ignore]
// Check if forward and backward probabilities would match
fn test_forward_vs_backward_probs() {
    // Create a mock sequence
    let (model, 
        (init_dist_hat, 
        trans_mat_hat, 
        emit_mat_hat)
    ) = common::case2();

    let obs = model.observations;
    let iteration = 3;

    let mut hmm = HMM::new(
        obs,
        init_dist_hat,
        trans_mat_hat,
        emit_mat_hat,
        iteration
    );

    // calculate regular forward & backforward probabilities
    hmm.train();
    hmm._update_proba_seq_given_model();
    let forward_proba = hmm.proba_seq_given_model;
    hmm._update_proba_seq_given_model_beta();
    let backward_proba = hmm.proba_seq_given_model;

    
    println!("Forward vs backward probability: {:.10e} vs {:.10e}", eexpo(forward_proba), eexpo(backward_proba));

}


// #[test]
// #[ignore]
// fn test_viterbi() {
//     // thinking coin flip with a normal (state 0) and fixed coin (state 1)
//     // assume observation 1 = head, 0 = tail
//     let (obs, 
//         init_dist, 
//         trans_mat, 
//         emit_mat
//     ) = common::case3();

//     let (v, b) = hmm::viterbi(
//          &obs, 
//          &init_dist, 
//          &trans_mat, 
//          &emit_mat);
    
//     let mut best_path_prob = 0.0;
//     let mut best_bp = 0;
//     for (i, row) in v.axis_iter(Axis(0)).enumerate() {
//         let prob_temp = row[obs.len() - 1];
//         if  prob_temp > best_path_prob {
//             best_path_prob = prob_temp;
//             best_bp = i;
//         }
//     }

//     let bestpath = hmm::traceback_viterbi(&b, best_bp);

//     println!("Viterbi Matrix with shape{:?}\n{:?}", v.shape(), v);
//     println!("Backtrace Matrix with shape{:?}\n{:?}", b.shape(), b);

//     println!("Viterbi probability = {:?}", best_path_prob);
//     println!("Best Path = {:?}", bestpath);


// }


