use rusty_hmm::hmm;
use ndarray::{Axis, arr2};


#[test]
// #[ignore]
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