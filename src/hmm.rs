//
//  Hidden Markov Model implementation following Speech and Language Processing. 
//  Daniel Jurafsky & James H. Martin. Copyright c 2019 (Draft of October 2, 2019).
//
//


// use std::collections::HashMap;
use std::vec::Vec;
use ndarray::{Array2, Array3, Axis, s};


const CONVERGENCE_TOLERANCE: f64 = 0.00000001;
const MAX_ITERATION: u32 = 100;


/// Calculate forward probability
///
/// INPUTS:
/// obs: sequence of observations represented as indices in emit_mat
/// init_dist: initial distribuition of states. it assumes its index corresponds to row index of trans_mat and emit_mat (for the former, both row & column indices)
/// trans_mat: state transition matrix
/// emit_mat: emission matrix. sates along the row, possible observations along the column
pub fn forward(obs:&Vec<u8>, init_dist: &Vec<f64>, trans_mat: &Array2<f64>, emit_mat: &Array2<f64>) -> Array2<f64> {
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
        // println!("FOWARD-PROCESSING {:?}/{:?} OBSERVATIONS", ind_obs + 1, len_obs);
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

    forward_mat
}

pub fn get_forward_prob(forward_mat: &Array2<f64>) -> f64 {
    let forward_prob = forward_mat.sum_axis(Axis(0))[forward_mat.shape()[1]-1];
    forward_prob
}


pub fn backward(obs:&Vec<u8>, trans_mat: &Array2<f64>, emit_mat: &Array2<f64>) -> Array2<f64> {
    let len_obs = obs.len();                    // length of observations
    let num_states = trans_mat.shape()[0];      // number of possible states
    let mut backward_mat = Array2::<f64>::zeros((num_states, len_obs));  // backward matrix

    // if the shape doesn't match, raise exception
    if emit_mat.shape()[0] != num_states {
        panic!{"Number of rows in emit_mat does not match with that of trans_mat."}
    };

    // if sume along the row for emit_mat does not equal 1 { panic! }

    // initialize
    for ind_state in 0..num_states {
        backward_mat[[ind_state, len_obs - 1]] = 1.0;
    };


    for ind_obs in (0..(len_obs - 1)).rev() {                     // for each observation along the observations sequence
        // println!("BACKWARD-PROCESSING {:?}/{:?} REMANINING OBSERVATIONS", ind_obs + 1, len_obs);
        for ind_curr_state in 0..num_states {       // for each state
            // calculate the probability of seeing the observation obs[ind_obs] for state ind_current_state by summing the probabilities
            //
            let mut backward_temp = 0.0;             
            for ind_next_state in 0..num_states{
                let b_ = backward_mat[[ind_next_state, ind_obs + 1]] * trans_mat[[ind_curr_state, ind_next_state]] * emit_mat[[ind_next_state, obs[ind_obs+1] as usize]];
                backward_temp += b_;
            }

            backward_mat[[ind_curr_state, ind_obs]] = backward_temp;
        }
    }

    backward_mat
}

pub fn get_backward_prob(
    backward_mat: &Array2<f64>, 
    obs:&Vec<u8>, init_dist: &Vec<f64>, emit_mat: &Array2<f64>) -> f64 {
    
    let num_states = backward_mat.shape()[0];
    let mut backward_prob = 0.0;
    for ind_state in 0..num_states {
        backward_prob +=  backward_mat[[ind_state, 0]] * init_dist[ind_state] * emit_mat[[ind_state, obs[0] as usize]];
    }

    backward_prob
}


/// Estimation of transition and emission probability matrices with initial estimates
/// INPUTS:
/// 
pub fn forward_backward(
    obs:&Vec<u8>, init_dist: &Vec<f64>, trans_mat: &mut Array2<f64>, emit_mat: &mut Array2<f64>, max_iter:u32) -> (Array2<f64>, Array2<f64>) {
    
    if trans_mat.shape()[0] != trans_mat.shape()[1] {
        panic!("trans_mat must be a square matrix with the same number of rows and columns.")
    }
    if init_dist.len() != trans_mat.shape()[0] {
        panic!("Number of states in init_dist must be the same as the one in trans_mat.")
    }
    if emit_mat.shape()[0] != trans_mat.shape()[0] {
        panic!("Number of states in emit_mat must be the same as the one in trans_mat.")
    }

    let num_states = trans_mat.shape()[0];
    let num_obs = emit_mat.shape()[1];
    let len_obs = obs.len();

    let mut iter_counter = 0;       // counter for iteration
    let mut convergence = 1.0;      // tracking convergence

    // gamma[[j, t]] tracks probability of being in j-th state (along the row) at time t (along the colukn)
    let mut gamma = Array2::<f64>::zeros((num_states, len_obs));

    // xi[[i, j, t]] tracks probability of being in state i (1st dimenstion) at time t (3rd dimension) and state j (2nd dimension) at time t+1
    let mut xi = Array3::<f64>::zeros((num_states, num_states, len_obs));


    while (convergence > CONVERGENCE_TOLERANCE) && (iter_counter < max_iter) {
        println!("Iteration {:?}/{:?}", iter_counter + 1, max_iter);
        let alpha = forward(obs, init_dist, trans_mat, emit_mat);       // compute forward matrix
        let beta = backward(obs, trans_mat, emit_mat);                  // compute backword matrix
        let prob_obs_model = get_forward_prob(&alpha);      // Prob of obs given the model as forward prob of the whole utterance

        // compute gamma and xi
        for i in 0..num_states {
            for t in 0..len_obs {
                
                gamma[[i, t]] = alpha[[i, t]] * beta[[i, t]] / prob_obs_model;
                
                for j in 0..num_states {
                    if t != len_obs - 1 {
                        xi[[i, j, t]] = alpha[[i, t]] * trans_mat [[i, j]] * emit_mat[[j, obs[t+1] as usize]] * beta[[j, t + 1]] / prob_obs_model;
                    }

                }
            }
        }

        // re-estimate trans_mat
        for i in 0..num_states {
            for j in 0..num_states {

                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for t in 0..len_obs {

                    numerator += xi[[i, j, t]];

                    for k in 0..num_states {
                        denominator += xi[[i, k, t]];
                    }
                };
                
                trans_mat[[i, j]] = numerator / denominator;

            }
        }


        // re-estimate emit_mat
        for i in 0..num_states {
            for o in 0..num_obs {

                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for t in 0..len_obs {
                    if obs[t] as usize == o {

                        numerator += gamma[[i, t]];

                    };

                    denominator += gamma[[i, t]];
                }
                
                emit_mat[[i, o]] = numerator / denominator;
            }
        }

        iter_counter += 1;

    };

    (trans_mat.to_owned(), emit_mat.to_owned())
}




/// Calculate viterbi probability
///
/// INPUTS:
/// obs: sequence of observations represented as indices in emit_mat
/// init_dist: initial distribuition of states. it assumes its index corresponds to row index of trans_mat and emit_mat (for the former, both row & column indices)
/// trans_mat: state transition matrix
/// emit_mat: emission matrix. sates along the row, possible observations along the column
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


/// Traceback the backpointers on the backpointer matrix from Viterbi algorithm
/// INPUTS:
/// backpointer: Array2 containing the backpointers from Viterbi algo
/// start: the last backpointer from Viterbi algorithm.
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