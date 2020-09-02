use rand::Rng;
use ndarray::{Array2};
use std::vec::Vec;
use super::utility;

pub fn generate_state_sequence(init_dist: &Vec<f64>, trans_mat: &Array2<f64>, length: usize) -> Vec<usize> {
    if trans_mat.shape()[0] != trans_mat.shape()[1] {
        panic!("trans_mat must be a square matrix with the same number of rows and columns.")
    }
    if init_dist.len() != trans_mat.shape()[0] {
        panic!("Number of states in init_dist must be the same as the one in trans_mat.")
    }

    let mut sequence = Vec::<usize>::with_capacity(length);

    // pick first state in the sequence
    let first_pick = utility::pick_index_from_cumulative_prob_vector(
        &utility::calculate_vec_elementwise_cumulative_sum(&init_dist)
    );
    sequence.push(first_pick);

    for i in 1..length {
        let prev_pick = sequence[i-1];
        let curr_pick = utility::pick_index_from_cumulative_prob_vector(
            &utility::calculate_array2_elementwise_cumulative_sum(&trans_mat, prev_pick)
        );
        sequence.push(curr_pick);
    }

    sequence

}

pub fn generate_observation_sequence(state_sequence: &Vec<usize>, emit_mat: &Array2<f64>) -> Vec<usize> {
    let mut sequence = Vec::<usize>::with_capacity(state_sequence.len());

    for state in state_sequence.iter() {
        let curr_obs = utility::pick_index_from_cumulative_prob_vector(
            &utility::calculate_array2_elementwise_cumulative_sum(
                &emit_mat, *state
            )
        );
        sequence.push(curr_obs);
    }

    sequence

}