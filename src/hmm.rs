use std::collections::HashMap;
use std::vec::Vec;
use ndarray::{Array2, s, Axis};

/// Calculate forward probability
///
/// INPUTS:
/// obs: sequence of observations represented as indices in emit_mat
/// init_dist: initial distribuition of states. it assumes its index corresponds to row index of trans_mat and emit_mat (for the former, both row & column indices)
/// trans_mat: state transition matrix
/// emit_mat: emission matrix. sates along the row, possible observations along the column
pub fn forward(obs:&Vec<u8>, init_dist: &Vec<f64>, trans_mat: &Array2<f64>, emit_mat: &Array2<f64>) -> Array2<f64> /*f64*/ {
    let len_obs = obs.len();           // length of observations
    let num_states = trans_mat.shape()[0];      // number of possible states
    let mut forward_mat = Array2::<f64>::zeros((num_states, len_obs));  // forward matrix

    // if the shape doesn't match, raise exception
    if emit_mat.shape()[0] != num_states {
        panic!{"Number of rows in emit_mat does not match with that of trans_mat."}
    };

    // if sume along the row for emit_mat does not equal 1 { panic! }

    // initialize
    for ind_state in 0..num_states {
        forward_mat[[ind_state, 0]] = init_dist[ind_state] * emit_mat[[ind_state, obs[0] as usize]];
    };


    for ind_obs in 1..len_obs {                     // for each observation along the observations sequence
        for ind_curr_state in 0..num_states {       // for each state
            // calculate the probability of seeing the observation obs[ind_obs] for state ind_current_state by summing the probabilities
            // of coming from each potential path.
            let mut forward_temp = 0.0;             
            for ind_prev_state in 0..num_states{
                let f = forward_mat[[ind_prev_state, ind_obs-1]] * trans_mat[[ind_prev_state, ind_curr_state]] * emit_mat[[ind_curr_state, obs[ind_obs] as usize]];
                forward_temp += f;
            }

            forward_mat[[ind_curr_state, ind_obs]] = forward_temp;
        }
    }

    let forward = forward_mat.sum_axis(Axis(0))[len_obs - 1];

    // forward
    forward_mat
}


struct HMM {
    states: Vec<String>,                            // set of state names
    // state_string2index_map: HashMap<String, u8>,    // hashmap to convert state names to numerical index
    // obs_string2index_map: HashMap<String, u8>,      // hashmap to convert observation names to numerical index

    init_state_distribution: Vec<f64>,              // Initial distribution of states
    trans_mat: Array2<f64>,                         // state transition probability matrix
    emit_mat: Array2<f64>,                          // emission probability matrix
}

impl HMM {

    pub fn new() -> HMM {
        HMM {
            states: Vec::<String>::new(),
            // state_string2index_map = HashMap::new(),
            init_state_distribution: Vec::<f64>::new(),
            trans_mat: Array2::<f64>::zeros((0, 0)),
            emit_mat: Array2::<f64>::zeros((0, 0)),
        }
    }



}