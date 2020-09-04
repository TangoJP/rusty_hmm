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
pub fn pick_index_from_cumulative_prob_vector(cumu_prob_vector: &Vec<f64>) -> usize {
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



// extended exponential function to deal with NAN
pub fn eexpo(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.exp()
    }
}

// extended natural log function to deal with NAN
pub fn eln(x: f64) -> f64 {
    if x > 0.0 {
        x.ln()
    } else if x == 0.0 {
        f64::NAN
    } else {
        panic!("Invalid f64 number entered for the function (e.g. negative number).")
    }
}

// takes in extended_ln (above) of two numbers and returns extended_ln(x + y)
pub fn eln_sum(eln_x: f64, eln_y:f64) -> f64 {
    if  eln_x.is_nan() || eln_y.is_nan() {
        if eln_x.is_nan() {
            eln_y
        } else {
            eln_x
        }
    } else {
        if eln_x > eln_y {
            eln_x + eln(1.0 + (eln_y - eln_x).exp())
        } else {
            eln_y + eln(1.0 + (eln_x - eln_y).exp())
        }
    }
}

// takes in extended_ln (above) of two numbers and returns extended_ln(xy), i.e. sum of inputs
pub fn eln_product(eln_x: f64, eln_y:f64) -> f64 {
    if  eln_x.is_nan() || eln_y.is_nan() {
        f64::NAN
    } else {
        eln_x + eln_y
    }
}




