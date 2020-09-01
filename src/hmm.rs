// use std::collections::HashMap;
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



pub fn viterbi(obs:&Vec<u8>, init_dist: &Vec<f64>, trans_mat: &Array2<f64>, emit_mat: &Array2<f64>) -> (Array2<f64>, Array2<usize>) /*f64*/ {
    let len_obs = obs.len();                    // length of observations
    let num_states = trans_mat.shape()[0];      // number of possible states
    let mut v_mat = Array2::<f64>::zeros((num_states, len_obs));  // viterbi matrix
    let mut bp_mat = Array2::<usize>::ones((num_states, len_obs));  // backpointer matrix

    // if the shape doesn't match, raise exception
    if emit_mat.shape()[0] != num_states {
        panic!{"Number of rows in emit_mat does not match with that of trans_mat."}
    };

    // if sume along the row for emit_mat does not equal 1 { panic! }

    // initialize
    for ind_state in 0..num_states {
        v_mat[[ind_state, 0]] = init_dist[ind_state] * emit_mat[[ind_state, obs[0] as usize]];
        bp_mat[[ind_state, 0]] = 0;
    };


    for ind_obs in 1..len_obs {                     // for each observation along the observations sequence
        for ind_curr_state in 0..num_states {       // for each state
            // calculate the probability of seeing the observation obs[ind_obs] for state ind_current_state as the max probabilities
            // of coming from all potential paths.
            let mut vprob_temp = 0.0;
            let mut bp_ind_temp = 0;           
            for ind_prev_state in 0..num_states{
                let v_ = v_mat[[ind_prev_state, ind_obs-1]] * trans_mat[[ind_prev_state, ind_curr_state]] * emit_mat[[ind_curr_state, obs[ind_obs] as usize]];
                if v_ > vprob_temp {
                    vprob_temp = v_;
                    bp_ind_temp = ind_prev_state;
                }
            }

            v_mat[[ind_curr_state, ind_obs]] = vprob_temp;
            bp_mat[[ind_curr_state, ind_obs]] = bp_ind_temp;
        }
    }

    let mut best_path_prob = 0.0;
    let mut best_bp = 0;
    for (i, row) in v_mat.axis_iter(Axis(0)).enumerate() {
        let prob_temp = row[len_obs - 1];
        if  prob_temp > best_path_prob {
            best_path_prob = prob_temp;
            best_bp = i;
        }
    }


    // forward
    (v_mat, bp_mat)
}

pub fn traceback_viterbi(backpointer: &Array2<usize>, start: usize) -> Vec<usize> {
    let path_length = backpointer.shape()[1];
    let mut path = Vec::<usize>::with_capacity(path_length);

    let mut i = path_length - 1;
    let mut j = start;
    path.push(start);
    while i > 0 {
        let prev_state = backpointer[[j, i]];
        path.push(prev_state);
        j = prev_state;
        i -= 1;
    }

    path.reverse();

    path

}


// struct HMM {
//     states: Vec<String>,                            // set of state names
//     // state_string2index_map: HashMap<String, u8>,    // hashmap to convert state names to numerical index
//     // obs_string2index_map: HashMap<String, u8>,      // hashmap to convert observation names to numerical index

//     init_state_distribution: Vec<f64>,              // Initial distribution of states
//     trans_mat: Array2<f64>,                         // state transition probability matrix
//     emit_mat: Array2<f64>,                          // emission probability matrix
// }

// impl HMM {

//     pub fn new() -> HMM {
//         HMM {
//             states: Vec::<String>::new(),
//             // state_string2index_map = HashMap::new(),
//             init_state_distribution: Vec::<f64>::new(),
//             trans_mat: Array2::<f64>::zeros((0, 0)),
//             emit_mat: Array2::<f64>::zeros((0, 0)),
//         }
//     }



// }