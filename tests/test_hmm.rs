use rusty_hmm::hmm::HMM;
use rusty_hmm::utility::{check_prob_matrix_sums_to_one, check_prob_vector_sums_to_one};
use ndarray::{Axis, arr2};

mod common;

#[test]
#[ignore]
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

// ⎄#[test]
// #[ignore]
// fn test_forward_backward() {
//     let (model, 
//         (init_dist_hat, 
//         mut trans_mat_hat, 
//         mut emit_mat_hat)
//     ) = common::case2();
    
//     let obs = model.observations;
//     let iteration = 100;

//     let (a_hat, b_hat) = hmm::forward_backward(
//         &obs, 
//         &init_dist_hat, 
//         &mut trans_mat_hat, 
//         &mut emit_mat_hat, 
//         iteration);
    
//     println!("Actual trans_mat\n{:?}", model.trans_mat);
//     println!("Estimated trans_mat\n{:?}", a_hat);

//     println!("Actual emit_mat\n{:?}", model.emit_mat);
//     println!("Estimated emit_mat\n{:?}", b_hat);
    
// }


// #[test]
// #[ignore]
// // Check if forward and backward probabilities would match
// fn test_regular_vs_log_probs() {
//     // Create a mock sequence
//     let (model, _) = common::case2();
//     let obs = model.observations;
    
//     // calculate regular forward & backforward probabilities
//     let forward_prob = hmm::get_forward_prob(
//         &hmm::forward(&obs, &model.init_dist, &model.trans_mat, &model.emit_mat)
//     );
//     let backward_prob = hmm::get_backward_prob(
//         &hmm::backward(&obs, &model.trans_mat, &model.emit_mat),
//         &obs, &model.init_dist, &model.emit_mat
//     );

//     let log_forward_prob = log_hmm::log_compute_forward_prob(
//         &log_hmm::log_compute_forward_matrix(&obs, &model.init_dist, &model.trans_mat, &model.emit_mat)
//     );

//     println!("Computing log_backward_prob");
//     let log_backward_prob = log_hmm::log_compute_backward_prob(
//         &log_hmm::log_compute_backward_matrix(&obs, &model.trans_mat, &model.emit_mat),
//         &obs, &model.init_dist, &model.emit_mat
//     );


//     println!("Forward  probability (ln(regular) vs ln_prob): {:?} vs {:?}", forward_prob.ln(), log_forward_prob);
//     println!("Backward probability (ln(regular) vs ln_prob): {:?} vs {:?}", backward_prob.ln(), log_backward_prob);

// }


// #[test]
// #[ignore]
// // Check if forward and backward probabilities would match
// fn test_forward_and_backward_probs() {
//     // Create a mock sequence
//     let (model, _) = common::case2();
//     let obs = model.observations;

//     let forward_prob = hmm::get_forward_prob(
//         &hmm::forward(&obs, &model.init_dist, &model.trans_mat, &model.emit_mat)
//     );
//     let backward_prob = hmm::get_backward_prob(
//         &hmm::backward(&obs, &model.trans_mat, &model.emit_mat),
//         &obs, &model.init_dist, &model.emit_mat
//     );

//     println!("Forward  probability = {:?}", forward_prob.log10());
//     println!("Backward probability = {:?}", backward_prob.log10());

// }

// #[test]
// #[ignore]
// fn test_forward() {
//     // thinking coin flip with a normal (state 0) and fixed coin (state 1)
//     // assume observation 1 = head, 0 = tail
//     let obs = vec![
//         0usize, 1usize, 1usize, 1usize, 0usize, 0usize, 0usize, 1usize, 
//         1usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 
//         0usize, 0usize
//     ];
//     let mut init_dist = Vec::<f64>::new();
//     init_dist.push(0.65);
//     init_dist.push(0.35);

//     let trans_mat = arr2(&[
//         [0.6, 0.4],
//         [0.4, 0.6]
//     ]);
//     let emit_mat = arr2(&[
//         [0.5, 0.5],
//         [0.8, 0.2]
//     ]);

//     let forward_mat = hmm::forward(
//          &obs, 
//          &init_dist, 
//          &trans_mat, 
//          &emit_mat);
    
//     let forward_prob = forward_mat.sum_axis(Axis(0))[obs.len() - 1];

//     println!("Forward Matrix with shape{:?}\n{:?}", forward_mat.shape(), forward_mat);
//     println!("Forward probability = {:?}", forward_prob);

// }

// #[test]
// #[ignore]
// fn test_backward() {
//     // thinking coin flip with a normal (state 0) and fixed coin (state 1)
//     // assume observation 1 = head, 0 = tail
//     let (obs, 
//         init_dist, 
//         trans_mat, 
//         emit_mat
//     ) = common::case3();

//     let backward_mat = hmm::backward(
//          &obs, 
//          &trans_mat, 
//          &emit_mat);
    
//     let backward_prob = hmm::get_backward_prob(
//     &backward_mat, 
//     &obs,
//     &init_dist,
//     &emit_mat
//     );

//     println!("Backward Matrix with shape{:?}\n{:?}", backward_mat.shape(), backward_mat);
//     println!("Backward probability = {:?}", backward_prob);

// }

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


