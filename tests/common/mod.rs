use rusty_hmm::generative::MockHMMModel;
use ndarray::{arr2, Array2};

pub fn case1() -> (MockHMMModel, (Vec<f64>, Array2<f64>, Array2<f64>)){
    // Create test sequence
    let mut model = MockHMMModel::new(
        200000,
        2,
        2,
    );

    model.set_init_dist_vector(vec![0.5, 0.5]);
    model.set_transition_matrix(
        arr2(&[
            [0.7, 0.3],
            [0.3, 0.7]
        ])
    );
    model.set_emission_matrix(
        arr2(&[
            [0.01, 0.99],
            [0.99, 0.01]
        ])
    );

    model.generate_sequence();

    let init_dist = vec![0.5, 0.5];
    let trans_mat = arr2(&[
        [0.7, 0.3],
        [0.3, 0.7]
    ]);
    let emit_mat = arr2(&[
        [0.7, 0.3],
        [0.3, 0.7]
    ]);
    
    let initial_condition = (init_dist, trans_mat, emit_mat);

    (model, initial_condition)

}


pub fn case2() -> (MockHMMModel, (Vec<f64>, Array2<f64>, Array2<f64>)){
    // Create test sequence
    let mut model = MockHMMModel::new(
        200000,
        3,
        4,
    );

    model.set_init_dist_vector(vec![0.34, 0.33, 0.33]);
    model.set_transition_matrix(
        arr2(&[
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
    );
    model.set_emission_matrix(
        arr2(&[
            [0.4, 0.4, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7]
        ])
    );

    model.generate_sequence();

    let init_dist = vec![0.34, 0.33, 0.33];
    let trans_mat = arr2(&[
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]
    ]);
    let emit_mat = arr2(&[
        [0.25, 0.25, 0.25, 0.25],
        [0.4, 0.1, 0.4, 0.1],
        [0.1, 0.4, 0.1, 0.4]
    ]);
    
    let initial_condition = (init_dist, trans_mat, emit_mat);

    (model, initial_condition)

}


pub fn case3() -> (Vec<usize>, Vec<f64>, Array2<f64>, Array2<f64>) {
    // thinking coin flip with a normal (state 0) and fixed coin (state 1)
    // assume observation 1 = head, 0 = tail
    let obs = vec![
        0usize, 1usize, 1usize, 1usize, 0usize, 0usize, 0usize, 1usize, 
        1usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 
        0usize, 0usize
    ];
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

    (obs, init_dist, trans_mat, emit_mat)
}