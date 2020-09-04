//
//  Hidden Markov Model implementation following Speech and Language Processing. 
//  Daniel Jurafsky & James H. Martin. Copyright c 2019 (Draft of October 2, 2019).
//
//  Log calculation version of hmm.rs following 
//  http://bozeman.genome.washington.edu/compbio/mbt599_2006/hmm_scaling_revised.pdf
//
//


const LOGZERO: f64 = f64::NAN;

// use std::collections::HashMap;
use std::vec::Vec;
use ndarray::{Array2, Array3, Axis, s};
use super::utility::{eexpo, eln, eln_sum, eln_product};

const CONVERGENCE_TOLERANCE: f64 = 0.00000001;
const MAX_ITERATION: u32 = 100;


/// Calculate log_forward probability
///
/// INPUTS:
/// obs: sequence of observations represented as indices in emit_mat
/// init_dist: initial distribuition of states. it assumes its index corresponds to row index of trans_mat and emit_mat (for the former, both row & column indices)
/// trans_mat: state transition matrix
/// emit_mat: emission matrix. sates along the row, possible observations along the column
pub fn log_compute_forward_matrix(obs:&Vec<u8>, init_dist: &Vec<f64>, trans_mat: &Array2<f64>, emit_mat: &Array2<f64>) -> Array2<f64> {
    let len_obs = obs.len();           // length of observations
    let num_states = trans_mat.shape()[0];      // number of possible states
    let mut log_forward_mat = Array2::<f64>::zeros((num_states, len_obs));  // log_forward matrix

    // if the shape doesn't match, raise exception
    if emit_mat.shape()[0] != num_states {
        panic!{"Number of rows in emit_mat does not match with that of trans_mat."}
    };

    // if sume along the row for emit_mat does not equal 1 { panic! }

    // initialize for the 1st time point
    for ind_state in 0..num_states {
        log_forward_mat[[ind_state, 0]] = eln_product(
            eln(init_dist[ind_state]), 
            eln(emit_mat[[ind_state, obs[0] as usize]])
        );
    };

    // recursively fill in the rest
    for ind_obs in 1..len_obs {                     // for each observation along the observations sequence
        // println!("forward-PROCESSING {:?}/{:?} OBSERVATIONS", ind_obs + 1, len_obs);
        for ind_curr_state in 0..num_states {       // for each state
            // calculate the probability of seeing the observation obs[ind_obs] for state ind_current_state by summing the probabilities
            // of coming from each potential path.
            let mut log_forward_temp = LOGZERO;                     // place holder
            for ind_prev_state in 0..num_states{            // do multiplication, i.e. log sum, of log_forward_mat and trans_mat which depend on prev_state
                log_forward_temp = eln_sum(
                    log_forward_temp,
                    eln_product(
                        log_forward_mat[[ind_prev_state, ind_obs-1]],
                        eln(trans_mat[[ind_prev_state, ind_curr_state]])
                    )
                );
            }
            log_forward_temp = eln_product(                 // do log multiplication with emit_mat because it only depends on current state
                log_forward_temp, 
                eln(emit_mat[[ind_curr_state, obs[ind_obs] as usize]])
            );

            log_forward_mat[[ind_curr_state, ind_obs]] = log_forward_temp;
        }
    }

    log_forward_mat
}


pub fn log_compute_forward_prob(log_forward_mat: &Array2<f64>) -> f64 {
    println!("Computing log_forward_prob");
    let mut log_forward_prob = LOGZERO;
    for i in 0..log_forward_mat.shape()[0] {
        log_forward_prob = eln_sum(
            log_forward_prob, 
            log_forward_mat[[i, log_forward_mat.shape()[1]-1]]
        );
    }
    log_forward_prob
}


pub fn log_compute_backward_matrix(obs:&Vec<u8>, trans_mat: &Array2<f64>, emit_mat: &Array2<f64>) -> Array2<f64> {
    let len_obs = obs.len();                    // length of observations
    let num_states = trans_mat.shape()[0];      // number of possible states
    let mut log_backward_mat = Array2::<f64>::zeros((num_states, len_obs));  // log_backward matrix

    // if the shape doesn't match, raise exception
    if emit_mat.shape()[0] != num_states {
        panic!{"Number of rows in emit_mat does not match with that of trans_mat."}
    };

    // if sume along the row for emit_mat does not equal 1 { panic! }

    // initialize
    for ind_state in 0..num_states {
        log_backward_mat[[ind_state, len_obs - 1]] = 0.0;
    };


    for ind_obs in (0..(len_obs - 1)).rev() {                     // for each observation along the observations sequence
        // println!("log_backward-PROCESSING {:?}/{:?} REMANINING OBSERVATIONS", ind_obs + 1, len_obs);
        for ind_curr_state in 0..num_states {       // for each state
            // calculate the probability of seeing the observation obs[ind_obs] for state ind_current_state by summing the probabilities
            //
            let mut log_backward_temp = LOGZERO;             
            for ind_next_state in 0..num_states{
                log_backward_temp = eln_sum(
                    log_backward_temp, 
                    eln_product(
                        eln(trans_mat[[ind_curr_state, ind_next_state]]),
                        eln_product(
                            eln(emit_mat[[ind_next_state, obs[ind_obs+1] as usize]]), 
                            log_backward_mat[[ind_next_state, ind_obs + 1]])
                    )
                );
            }

            log_backward_mat[[ind_curr_state, ind_obs]] = log_backward_temp;
        }
    }

    log_backward_mat
}


pub fn loc_compute_backward_prob(log_backward_mat: &Array2<f64>, obs:&Vec<u8>, init_dist: &Vec<f64>, emit_mat: &Array2<f64>) -> f64 {
    
    let mut log_backward_prob = LOGZERO;
    for ind_state in 0..log_backward_mat.shape()[0] {
        log_backward_prob = eln_sum(
            log_backward_prob, 
            eln_product(
                eln(init_dist[ind_state]), 
                eln_product(
                    log_backward_mat[[ind_state, 0]], 
                    eln(emit_mat[[ind_state, obs[0] as usize]])
                )
            )
        );
    }
    log_backward_prob
}

// compute the probability of being in state i at time t given the model and the observation sequene
pub fn log_compute_gamma(obs:&Vec<u8>, log_forward_mat: &Array2<f64>, log_backward_mat: &Array2<f64>) -> Array2<f64> {
    if log_forward_mat.shape() != log_backward_mat.shape() {
        panic!("log_forward_mat and log_backward_mat must be of the same shape.")
    };

    let num_states = log_forward_mat.shape()[0];
    let len_obs = obs.len();
    let mut log_gamma = Array2::<f64>::zeros((num_states, len_obs));
    let mut denominator: f64;

    // iterate over observation sequence
    for t in 0..len_obs {
        denominator = LOGZERO;
        
        // calculate numerator for each i and add each i-th value to denominator
        for i in 0..num_states {
            log_gamma[[i, t]] = eln_product(
                log_forward_mat[[i, t]], 
                log_backward_mat[[i, t]]
            );
            denominator = eln_sum(
                denominator, 
                log_gamma[[i, t]]);
        };

        // normalize the log_gamma value for obs = t
        for i in 0..num_states {
            log_gamma[[i, t]] = eln_product(log_gamma[[i, t]], -denominator);
        } 

    }
    log_gamma
}


// log_xi[[i, j, t]] tracks log probability of being in state i (1st dimenstion) at time t (3rd dimension) and state j (2nd dimension) at time t+1
pub fn log_compute_xi(obs:&Vec<u8>, trans_mat: &mut Array2<f64>, emit_mat: &mut Array2<f64>, log_forward_mat: &Array2<f64>, log_backward_mat: &Array2<f64>) -> Array3<f64> {

    if log_forward_mat.shape() != log_backward_mat.shape() {
        panic!("log_forward_mat and log_backward_mat must be of the same shape.")
    };
    if trans_mat.shape()[0] != trans_mat.shape()[1] {
        panic!("trans_mat must be a square matrix with the same number of rows and columns.")
    };
    if emit_mat.shape()[0] != trans_mat.shape()[0] {
        panic!("Number of states in emit_mat must be the same as the one in trans_mat.")
    };

    let num_states = log_forward_mat.shape()[0];
    let len_obs = obs.len();
    let mut log_xi = Array3::<f64>::zeros((num_states, num_states, len_obs));
    let mut denominator: f64;

    // iterate over observation sequence
    for t in 0..(len_obs-1) {
        denominator = LOGZERO;

        // calculate numerator for each i -> j transition and add value to denominator
        for i in 0..num_states {
            for j in 0..num_states {
                log_xi[[i, j, t]] = eln_product(
                    log_forward_mat[[i, t]],
                    log_backward_mat[[i, t+1]]
                );
                denominator = eln_sum(
                    denominator,
                    log_xi[[i, j, t]]
                );
            }
        };

        // normalize the log_gamma value for obs = t
        for i in 0..num_states {
            for j in 0..num_states {
                log_xi[[i, j, t]] = eln_product(
                    log_xi[[i, j, t]],
                    -denominator
                )
            }
        }
    }

    log_xi    
}


// estimate initial distribution from log_gamma_mat
pub fn estimate_initial_dist(log_gamma_mat: &Array2<f64>) -> Vec<f64> {
    let length = log_gamma_mat.shape()[0];
    let mut init_dist = Vec::<f64>::with_capacity(length);

    for i in 0..length {
        init_dist.push(eexpo(log_gamma_mat[[i, 0]]));
    }

    init_dist
}


// estimate transition matrix
pub fn estimate_trans_mat(log_gamma_mat: &Array2<f64>, log_xi_mat: &Array3<f64>) -> Array2<f64> {
    let num_states = log_gamma_mat.shape()[0];
    let len_obs = log_xi_mat.shape()[2];
    let mut a_hat = Array2::<f64>::zeros((num_states, num_states));

    // iterate over each i->j transitions
    for i in 0..num_states {
        for j in 0..num_states {
            let mut numerator = LOGZERO;
            let mut denominator = LOGZERO;

            // for each i->j transition, calculate numerator and denominator (normalizer)
            for t in 0..(len_obs-1) {
                numerator = eln_sum(
                    numerator,
                    log_xi_mat[[i, j, t]]
                );
                denominator = eln_sum(
                    denominator,
                    log_gamma_mat[[i, t]]
                );
            };
            
            // normalize numerator with denominator and assign its exponent to a_hat[i, j]
            a_hat[[i, j]] = eexpo(
                eln_product(
                    numerator,
                    -denominator
                )
            )
        }
    }

    a_hat

}


// estimate emission matrix
pub fn estimate_emit_mat(log_gamma_mat: &Array2<f64>, log_xi_mat: &Array3<f64>, obs:&Vec<u8>, num_obs: usize) -> Array2<f64> {
    let num_states = log_gamma_mat.shape()[0];
    let len_obs = log_xi_mat.shape()[2];
    let mut b_hat = Array2::<f64>::zeros((num_states, num_obs));

    // iterate over each emissions from state j to observation k
    for k in 0..num_obs {
        for j in 0..num_states {
            let mut numerator = LOGZERO;
            let mut denominator = LOGZERO;

            for t in 0..len_obs {

                // sum observations from state j where value k was observed
                if (obs[t] as usize )== k {
                    numerator = eln_sum(
                        numerator,
                        log_gamma_mat[[j, t]]
                    );
                };

                // sum over all observation from state j
                denominator = eln_sum(
                    denominator,
                    log_gamma_mat[[j, t]]
                );
            };

            // estimate the prob of emitting k from state j
            b_hat[[j, k]] = eexpo(
                eln_product(
                    numerator,
                    -denominator
                )
            );

        }
        
    };

    b_hat

}


/// Estimation of transition and emission probability matrices with initial estimates
/// INPUTS:
/// 
pub fn log_compute_forward_backward(
    obs:&Vec<u8>, init_dist: &mut Vec<f64>, trans_mat: &mut Array2<f64>, emit_mat: &mut Array2<f64>, max_iter:u32) -> (Vec<f64>, Array2<f64>, Array2<f64>) {
    
    if trans_mat.shape()[0] != trans_mat.shape()[1] {
        panic!("trans_mat must be a square matrix with the same number of rows and columns.")
    }
    if init_dist.len() != trans_mat.shape()[0] {
        panic!("Number of states in init_dist must be the same as the one in trans_mat.")
    }
    if emit_mat.shape()[0] != trans_mat.shape()[0] {
        panic!("Number of states in emit_mat must be the same as the one in trans_mat.")
    }

    let num_obs = emit_mat.shape()[1];

    let mut iter_counter = 0;       // counter for iteration
    let mut convergence = 1.0;      // tracking convergence


    // iterate till convergence or max_iter reached
    while (convergence > CONVERGENCE_TOLERANCE) && (iter_counter < max_iter) {
        println!("Iteration {:?}/{:?}", iter_counter + 1, max_iter);

        // print!("Initial Dist ({:?}/{:?})\n{:?}\n", iter_counter + 1, max_iter, init_dist);
        // print!("Trans Mat ({:?}/{:?})\n{:?}\n", iter_counter + 1, max_iter, trans_mat);
        // print!("Emit Mat ({:?}/{:?})\n{:?}\n", iter_counter + 1, max_iter, emit_mat);
        // print!("\n");

        let eln_alpha = log_compute_forward_matrix(obs, init_dist, trans_mat, emit_mat);
        let eln_beta = log_compute_backward_matrix(obs, trans_mat, emit_mat);
        let log_gamma = log_compute_gamma(obs, &eln_alpha, &eln_beta);
        let log_xi = log_compute_xi(obs, trans_mat, emit_mat, &eln_alpha, &eln_beta);

        // println!("log_gamma shape {:?}", log_gamma.shape());
        // println!("  log_xi  shape {:?}", log_xi.shape());

        *init_dist = estimate_initial_dist(&log_gamma);
        *trans_mat = estimate_trans_mat(&log_gamma, &log_xi);
        *emit_mat = estimate_emit_mat(&log_gamma, &log_xi, obs, num_obs);

        iter_counter += 1;

    };

    (init_dist.to_owned(), trans_mat.to_owned(), emit_mat.to_owned())

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


    // log_forward
    (v_mat, bp_mat)
}


// /// Traceback the backpointers on the backpointer matrix from Viterbi algorithm
// /// INPUTS:
// /// backpointer: Array2 containing the backpointers from Viterbi algo
// /// start: the last backpointer from Viterbi algorithm.
// pub fn traceback_viterbi(backpointer: &Array2<usize>, start: usize) -> Vec<usize> {
//     let path_length = backpointer.shape()[1];
//     let mut path = Vec::<usize>::with_capacity(path_length);

//     let mut i = path_length - 1;
//     let mut j = start;
//     path.push(start);
//     while i > 0 {
//         let prev_state = backpointer[[j, i]];
//         path.push(prev_state);
//         j = prev_state;
//         i -= 1;
//     }

//     path.reverse();

//     path

// }


