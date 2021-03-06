//
//  Library of utility functions.
//
//  There are three kinds of utility functions implemented here:
//  1. functions used for random sampling
//  2. functions used to check for matrix integrity
//  3. functions used for natural log calculations
//
//


use rand::Rng;
use ndarray::{Array2, Axis};
use std::vec::Vec;
use float_cmp::approx_eq;

const FLOAT_CMP_EPSILON_TOLERANCE:f64 = 0.0001;

/// The following are the functions used for sampling
// Create a new vector with cumulative sum up to each element of the input vector.
pub fn calculate_vec_elementwise_cumulative_sum(vector: &Vec<f64>) -> Vec<f64> {

    let mut cumu_vector = Vec::<f64>::with_capacity(vector.len());  // create a new empty vec

    cumu_vector.push(vector[0]);                            // push first value

    for i in 1..vector.len() {
        cumu_vector.push(cumu_vector[i-1] + vector[i]);     // keep on adding new val to prev cumu val
    }

    cumu_vector

}

// Create a new vector with cumulative sum up to each element in a row in an array.
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
    let cumu_prob = cumu_prob_vector[cumu_prob_vector.len() - 1];
    if !approx_eq!(f64, cumu_prob, 1.0, epsilon=0.00001) {
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



/// The following are the functions used to check probability matrix integrity
// check if the values in a vector sums to 1.0 (float comp. tolerance = FLOAT_CMP_EPSILON_TOLERANCE)
pub fn check_prob_vector_sums_to_one(vector: &Vec<f64>) -> bool {
    let mut sum = 0.0;
    for e in vector.iter() {
        sum += e;
    }
    approx_eq!(f64, sum, 1.0, epsilon=FLOAT_CMP_EPSILON_TOLERANCE)
}

// check if the values in each row of a matrix sums to 1.0 (float comp. tolerance = FLOAT_CMP_EPSILON_TOLERANCE)
pub fn check_prob_matrix_sums_to_one(matrix: &Array2<f64>) -> bool {
    for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {
        if !approx_eq!(f64, row.sum(), 1.0, epsilon=FLOAT_CMP_EPSILON_TOLERANCE) {
            println!("{}-th row does not add up to 1.0. {:?}", i, matrix.sum_axis(Axis(1)));
            return false;
        };
    }
    return true;
}



/// The following are the functions used for natural log calculations
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




