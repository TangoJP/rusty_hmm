use rand::Rng;
use ndarray::{Array2};
use std::vec::Vec;

pub fn generate_state_sequence(init_dist: Vec<usize>, trans_mat: &Array2<f64>, length: usize) -> Vec<usize> {
    if trans_mat.shape()[0] != trans_mat.shape()[1] {
        panic!("trans_mat must be a square matrix with the same number of rows and columns.")
    }
    if init_dist.len() != trans_mat.shape()[0] {
        panic!("Number of states in init_dist must be the same as the one in trans_mat.")
    }

    let num_states = init_dist.len();
    let mut sequence = Vec::<usize>::with_capacity(length);


    for i in 0..length {
        


    }

    sequence

}