use ndarray::Array2;
use std::vec::Vec;
use super::utility;


pub struct MockHMMModel {
    pub len_observations: usize,
    pub num_states: usize,
    pub num_observations: usize,
    pub init_dist: Vec<f64>,
    pub trans_mat: Array2<f64>,
    pub emit_mat: Array2<f64>,
    pub state_sequence: Vec<usize>,
    pub observations: Vec<usize>,
    init_dist_set: bool,
    trans_mat_set: bool,
    emit_mat_set: bool,
}

impl MockHMMModel {
    /// instatntiate a new MockHMMSequence obj with dimension parameters and empty arrays and vectors
    pub fn new(len_observations: usize, num_states: usize, num_observations: usize) -> MockHMMModel {
        MockHMMModel {
            len_observations,
            num_states,
            num_observations,
            init_dist: Vec::<f64>::with_capacity(len_observations),
            trans_mat: Array2::<f64>::zeros((num_states, num_states)),
            emit_mat: Array2::<f64>::zeros((num_states, num_observations)),
            state_sequence: Vec::<usize>::with_capacity(len_observations),
            observations: Vec::<usize>::with_capacity(len_observations),
            init_dist_set: false,
            trans_mat_set: false,
            emit_mat_set: false,

        }
    }

    /// set init_dist
    pub fn set_init_dist_vector(&mut self, init_dist: Vec<f64>) {
        if !utility::check_prob_vector_sums_to_one(&init_dist) {
            panic!("trans_mat rows do not add up to 1.00.")
        };

        if self.num_states == init_dist.len() {
            self.init_dist = init_dist;
            self.init_dist_set = true;
        } else {
            panic!("_ input must be of length = len_observations.")
        }
    }

    /// set trans_mat
    pub fn set_transition_matrix(&mut self, trans_mat: Array2<f64>) {
        if !utility::check_prob_matrix_sums_to_one(&trans_mat) {
            panic!("trans_mat rows do not add up to 1.00.")
        };

        if self.trans_mat.shape() == trans_mat.shape() {
            self.trans_mat = trans_mat;
            self.trans_mat_set = true;
        } else {
            panic!("trans_mat input must be an Array2 of num_states x num_states dimensions.")
        };
    }

    /// set emit_mat
    pub fn set_emission_matrix(&mut self, emit_mat: Array2<f64>) {
        if !utility::check_prob_matrix_sums_to_one(&emit_mat) {
            panic!("emit_mat rows do not add up to 1.00.")
        };

        if self.emit_mat.shape() == emit_mat.shape() {
            self.emit_mat = emit_mat;
            self.emit_mat_set = true;
        } else {
            panic!("emit_mat input must be an Array2 of num_states x len_observations dimensions.")
        };
    }

    /// private function to generate state sequence
    pub fn _generate_state_sequence(&mut self) {
        if self.trans_mat.shape()[0] != self.trans_mat.shape()[1] {
            panic!("trans_mat must be a square matrix with the same number of rows and columns.")
        }
        if self.init_dist.len() != self.trans_mat.shape()[0] {
            panic!("Number of states in init_dist must be the same as the one in trans_mat.")
        }
        // pick first state in the sequence
        let first_pick = utility::pick_index_from_cumulative_prob_vector(
            &utility::calculate_vec_elementwise_cumulative_sum(&self.init_dist)
        );
        self.state_sequence.push(first_pick);
    
        for i in 1..self.len_observations {
            let prev_pick = self.state_sequence[i-1];
            let curr_pick = utility::pick_index_from_cumulative_prob_vector(
                &utility::calculate_array2_elementwise_cumulative_sum(&self.trans_mat, prev_pick)
            );
            self.state_sequence.push(curr_pick);
        }
    }

    /// private function to generate observation sequence
    pub fn _generate_observation_sequence(&mut self) {    
        for state in self.state_sequence.iter() {
            let curr_obs = utility::pick_index_from_cumulative_prob_vector(
                &utility::calculate_array2_elementwise_cumulative_sum(
                    &self.emit_mat, *state
                )
            );
            self.observations.push(curr_obs);
        }
    
    }

    /// generate the state and observations equence based on the model
    pub fn generate_sequence(&mut self) {
        if self.init_dist_set && self.trans_mat_set && self.emit_mat_set {
            self._generate_state_sequence();
            self._generate_observation_sequence();
        } else {
            panic!("init_dist, trans_mat, and emit_mat must be set with their respective setter functions before generating a sequence.")
        }
    }
}

