use rusty_hmm::hmm;
use ndarray::{Axis, arr2};

#[test]
// #[ignore]
fn test_forward() {
    // thinking coin flip with a normal (state 0) and fixed coin (state 1)
    // assume observation 1 = head, 0 = tail
    let obs = vec![0u8, 1u8, 1u8, 1u8, 0u8, 0u8, 0u8, 1u8, 0u8, 1u8, 0u8, 0u8, 1u8, 0u8, 1u8];
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