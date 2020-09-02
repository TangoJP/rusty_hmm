use rand::Rng;
use ndarray::{Array2};
use std::vec::Vec;

/// Create a new vector with cumulative sum up to each element of the input vector.
pub fn calculate_vec_elementwise_cumulative_sum(vector: &Vec<f64>) -> Vec<f64> {

    let mut cumu_vector = Vec::<f64>::with_capacity(vector.len());  // create a new empty vec

    cumu_vector.push(vector[0]);                            // push first value

    for i in 1..vector.len() {
        cumu_vector.push(cumu_vector[i-1] + vector[i]);     // keep on adding new val to prev cumu val
    }

    cumu_vector

}
/// Create a new vector with cumulative sum up to each element in a row in an array.
pub fn calculate_array2_elementwise_cumulative_sum(array: &Array2<f64>, ind_row: usize) -> Vec<f64> {

    let mut cumu_vector = Vec::<f64>::with_capacity(array.shape()[1]);  // create a new empty vec
    
    cumu_vector.push(array[[ind_row, 0]]);                          // push first value
    
    for i in 1..array.shape()[1] {                          
        cumu_vector.push(cumu_vector[i-1] + array[[ind_row, i]]);   // keep on adding new val to prev cumu val
    }

    cumu_vector

}

// Randomly pick a state (i.e. index in the original prob vector) using the cumulative prob vector.
pub fn pick_a_state_from_cumulative_prob_vector(cumu_prob_vector: &Vec<f64>) -> usize {
    if cumu_prob_vector[cumu_prob_vector.len() - 1] != 1.0 {
        panic!("Cumulative probability does not sum up to 1.0.")
    }

    let mut rng = rand::thread_rng();
    let random_number = rng.gen::<f64>();

    let mut i = 0usize;
    while i < cumu_prob_vector.len() {
        if random_number < cumu_prob_vector[i] {
            break;
        }
        i += 1;
    }

    i
}