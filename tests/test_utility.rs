use rusty_hmm::utility;
use std::collections::HashMap;
use ndarray::arr2;

#[test]
#[ignore]
fn test_check_prob_vector_sums_to_one() {
    let good_vec = vec![0.11, 0.19, 0.3, 0.4];
    let bad_vec = vec![0.11, 0.19, 0.3, 0.42];

    let good_res = utility::check_prob_vector_sums_to_one(&good_vec);
    let bad_res = utility::check_prob_vector_sums_to_one(&bad_vec);

    println!("{:?} => {:?}", good_vec, good_res);
    println!("{:?} => {:?}", bad_vec, bad_res);

}


#[test]
#[ignore]
fn test_check_prob_matrix_sums_to_one() {
    let good_mat = arr2(&[
        [0.11, 0.19, 0.3, 0.4],
        [0.0, 0.5, 0.4, 0.1],
        [0.291, 0.21, 0.34, 0.159]
    ]);
    let bad_mat = arr2(&[
        [0.11, 0.19, 0.3, 0.40001],
        [0.0, 0.5, 0.4, 0.1],
        [0.25, 0.31, 0.34, 0.1]
    ]);

    let good_res = utility::check_prob_matrix_sums_to_one(&good_mat);
    let bad_res = utility::check_prob_matrix_sums_to_one(&bad_mat);

    println!("{:?} => {:?}", good_mat, good_res);
    println!("{:?} => {:?}", bad_mat, bad_res);
}


#[test]
#[ignore]
fn test_calculate_elementwise_cumulative_sum() {
    let vector = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let new_vector = utility::calculate_vec_elementwise_cumulative_sum(&vector);

    println!("Old Vector: {:?}", vector);
    println!("New Vector: {:?}", new_vector);
}

#[test]
#[ignore]
fn test_pick_index_from_cumulative_prob_vector() {
    let prob_vector = vec![0.1, 0.25, 0.2, 0.05, 0.4];
    let cumu_prob_vector = utility::calculate_vec_elementwise_cumulative_sum(&prob_vector);
    
    let mut counter = HashMap::<usize, f64>::new();
    for i in 0..prob_vector.len() {
        counter.insert(i, 0.0);
    }

    let num_draws = 100000;
    for _ in 0..num_draws {
        let pick = utility::pick_index_from_cumulative_prob_vector(&cumu_prob_vector);
        // println!("Pick{:?} = State {:?}", j+1, pick);

        *counter.get_mut(&pick).unwrap() += 1.0;
        
    }

    for i in 0..prob_vector.len() {
        *counter.get_mut(&i).unwrap() /= num_draws as f64;
    }

    println!("Result from {} draws\n{:?}", num_draws, counter);

}