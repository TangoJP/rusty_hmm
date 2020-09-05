use rusty_hmm::generative::MockHMMModel;
use ndarray::arr2;

mod common;

#[test]
#[ignore]
fn test_mock_hmm_model() {

    let mut model = MockHMMModel::new(
        50,
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
            [0.99, 0.01],
            [0.01, 0.99]
        ])
    );

    model.generate_sequence();

    println!("State Sequence: {:?}", model.state_sequence);
    println!(" Observations : {:?}", model.observations);

}