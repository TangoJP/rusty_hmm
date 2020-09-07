//
//  Hidden Markov Model implementation following Speech and Language Processing. 
//  Daniel Jurafsky & James H. Martin. Copyright c 2019 (Draft of October 2, 2019).
//
//  Log calculation version of hmm.rs following 
//  http://bozeman.genome.washington.edu/compbio/mbt599_2006/hmm_scaling_revised.pdf
//
//


use std::vec::Vec;
use ndarray::{Array2, Array3, Axis};
use super::utility::{eexpo, eln, eln_sum, eln_product};

const LOGZERO: f64 = f64::NAN;
const CONVERGENCE_TOLERANCE: f64 = 0.00001;  


pub struct HMM {
    pub observations: Vec<usize>,
    pub len_obs: usize,
    pub num_states: usize,
    pub num_obs: usize,
    pub init_dist: Vec<f64>,
    pub trans_mat: Array2<f64>,
    pub emit_mat: Array2<f64>,
    pub proba_seq_given_model: f64,
    input_init_dist: Vec<f64>,
    input_trans_mat:Array2<f64>,
    input_emit_mat:Array2<f64>, 
    log_alpha: Array2<f64>,
    log_beta: Array2<f64>,
    log_gamma: Array2<f64>,
    log_xi: Array3<f64>,
    pub log_viterbi: Array2<f64>,
    backpointer_mat: Array2<usize>,
    best_backpointer: usize,
    pub best_state_sequence: Vec<usize>,
    max_iter: i32,
    
}

impl HMM {
    /// Create a new model instance
    pub fn new(observations: Vec<usize>, init_dist: Vec<f64>, trans_mat: Array2<f64>, emit_mat: Array2<f64>, max_iter: i32) -> HMM {
        // if the shapes don't match, raise exception
        if init_dist.len() != trans_mat.shape()[0] {
            panic!("# of states in trans_mat does not match with that of init_dist.")
        };
        if trans_mat.shape()[0] != trans_mat.shape()[1] {
            panic!("trans_mat must be a square matrix with the same number of rows and columns.")
        };
        if init_dist.len() != emit_mat.shape()[0] {
            panic!{"Number of rows in emit_mat does not match with that of trans_mat."}
        };

        let length = observations.len();
        let number_of_states = init_dist.len();
        HMM {
            observations,
            len_obs: length,
            num_states: number_of_states,
            num_obs: emit_mat.shape()[1],
            init_dist: init_dist.clone(),
            trans_mat: trans_mat.clone(),
            emit_mat: emit_mat.clone(),
            proba_seq_given_model: LOGZERO,
            input_init_dist: init_dist,
            input_trans_mat: trans_mat,
            input_emit_mat: emit_mat,
            log_alpha: Array2::<f64>::zeros((number_of_states, length)),
            log_beta: Array2::<f64>::zeros((number_of_states, length)),
            log_gamma: Array2::<f64>::zeros((number_of_states, length)),
            log_xi: Array3::<f64>::zeros((number_of_states, number_of_states, length)),
            log_viterbi: Array2::<f64>::zeros((number_of_states, length)),
            backpointer_mat: Array2::<usize>::ones((number_of_states, length)),
            best_backpointer: 0,
            best_state_sequence: Vec::<usize>::with_capacity(length),
            max_iter
        }
    }

    /// update the forward matrix in place
    pub fn update_alpha(&mut self) {
        // initialize for the 1st time point
        for ind_state in 0..self.num_states {
            self.log_alpha[[ind_state, 0]] = eln_product(
                eln(self.init_dist[ind_state]), 
                eln(self.emit_mat[[ind_state, self.observations[0]]])
            );
        };

        // recursively fill in the rest
        for ind_obs in 1..self.len_obs {                    // for each observation along the observations sequence

            for ind_curr_state in 0..self.num_states {      // for each state
                // calculate the probability of seeing the observation 
                // obs[ind_obs] for state ind_current_state by summing the 
                // probabilities of coming from each potential path.
                let mut log_forward_temp = LOGZERO;                 // place holder
                for ind_prev_state in 0..self.num_states{       // do multiplication, i.e. log sum, of log_forward_mat and trans_mat which depend on prev_state
                    log_forward_temp = eln_sum(
                        log_forward_temp,
                        eln_product(
                            self.log_alpha[[ind_prev_state, ind_obs-1]],
                            eln(self.trans_mat[[ind_prev_state, ind_curr_state]])
                        )
                    );
                }
                log_forward_temp = eln_product(         // do log multiplication with emit_mat because it only depends on current state
                    log_forward_temp, 
                    eln(self.emit_mat[[ind_curr_state, self.observations[ind_obs]]])
                );

                self.log_alpha[[ind_curr_state, ind_obs]] = log_forward_temp;
            };
        };
    }

    /// update the backward matrix
    pub fn update_beta(&mut self) {
        // initialize
        for ind_state in 0..self.num_states {
            self.log_beta[[ind_state, self.len_obs - 1]] = 0.0;
        };

        for ind_obs in (0..(self.len_obs - 1)).rev() {       // for each observation along the observations sequence
            for ind_curr_state in 0..self.num_states {       // for each state
                // calculate the probability of seeing the observation obs[ind_obs]
                //  for state ind_current_state by summing the probabilities
                let mut log_backward_temp = LOGZERO;             
                for ind_next_state in 0..self.num_states{
                    log_backward_temp = eln_sum(
                        log_backward_temp, 
                        eln_product(
                            eln(self.trans_mat[[ind_curr_state, ind_next_state]]),
                            eln_product(
                                eln(self.emit_mat[[ind_next_state, self.observations[ind_obs+1]]]), 
                                self.log_beta[[ind_next_state, ind_obs + 1]])
                        )
                    );
                }

                self.log_beta[[ind_curr_state, ind_obs]] = log_backward_temp;
            };
        };
    }

    /// train the model with give inputs
    pub fn train(&mut self) {
        // iterate till convergence or max_iter reached
        let mut iter_counter = 0;           // counter for iteration
        let mut prev_score: f64 = 1.0;          // variable to hold previous prob of the sequence given the model
        while iter_counter < self.max_iter {     // iterate till max_iter reached, unless converges

            self.update_alpha();
            self.update_beta();
            self._update_proba_seq_given_model();
            
            self._compute_gamma();
            self._compute_xi();

            self._update_init_dist();
            self._update_trans_mat();
            self._update_emit_mat();


            println!("Processed iteration {:?}/{:?}: log-score = {:?}", 
                iter_counter + 1, 
                self.max_iter, 
                self.proba_seq_given_model
            );

            // compute change from the previous round and check for convergence
            if iter_counter != 0 {
                // 1-|curr_score - prev_score|/curr_score
                let change = 1.0 - eexpo((self.proba_seq_given_model - prev_score).abs());
                if change.abs() < CONVERGENCE_TOLERANCE {
                    break;
                }
            }

            prev_score = self.proba_seq_given_model;
            iter_counter += 1;
            
        };

        self._viterbi();
        self._traceback_viterbi();

        println!("Iteration finished at iteration {:?}/{:?}: final log-score = {:?}", iter_counter+1, self.max_iter, self.proba_seq_given_model);
        
    }

    /// update the probability of the observations given the current model
    pub fn _update_proba_seq_given_model(&mut self) {
        self.proba_seq_given_model = LOGZERO;
        for i in 0..self.log_alpha.shape()[0] {
            self.proba_seq_given_model = eln_sum(
                self.proba_seq_given_model, 
                self.log_alpha[[i, self.log_alpha.shape()[1]-1]]
            );
        };
    }

    /// update the probability of the observations given the current model using backward matrix
    pub fn _update_proba_seq_given_model_beta(&mut self) {
        self.proba_seq_given_model  = LOGZERO;
        for ind_state in 0..self.num_states {
            self.proba_seq_given_model = eln_sum(
                self.proba_seq_given_model, 
                eln_product(
                    eln(self.init_dist[ind_state]), 
                    eln_product(
                        self.log_beta[[ind_state, 0]], 
                        eln(self.emit_mat[[ind_state, self.observations[0]]])
                    )
                )
            );
        };
    }

    // compute log_gamma matrix
    fn _compute_gamma(&mut self) {
        // iterate over observation sequence
        for t in 0..self.len_obs {

            // calculate numerator for each i and add each i-th value to denominator
            for i in 0..self.num_states {
                self.log_gamma[[i, t]] = eln_product(
                    self.log_alpha[[i, t]], 
                    self.log_beta[[i, t]]
                );
                self.log_gamma[[i, t]] = eln_product(
                    self.log_gamma[[i, t]], 
                    -self.proba_seq_given_model
                );
            };
    
    
        }
    
    }

    // compute log_xi matrix
    fn _compute_xi(&mut self) {
        // iterate over observation sequence
        for t in 0..(self.len_obs-1) {

            // calculate numerator for each i -> j transition and add value to denominator
            for i in 0..self.num_states {
                for j in 0..self.num_states {
                    let component1 = eln_product(
                        self.log_alpha[[i, t]],
                        self.trans_mat[[i, j]]
                    );
                    let component2 = eln_product(
                        self.emit_mat[[j, self.observations[t+1]]],
                        self.log_beta[[j, t+1]]
                    );
                    self.log_xi[[i, j, t]] = eln_product(
                        component1, 
                        component2      
                    );
                    self.log_xi[[i, j, t]] = eln_product(
                        self.log_xi[[i, j, t]],
                        -self.proba_seq_given_model
                    );
                };
            };
        };
    }
    
    // update init_dist estimate
    fn _update_init_dist(&mut self) {
        self.init_dist.clear();
        for i in 0..self.num_states {
            self.init_dist.push(eexpo(self.log_gamma[[i, 0]]));
        }
    }

    // update trans_mat estimate
    fn _update_trans_mat(&mut self) {
        // iterate over each i->j transitions
        for i in 0..self.num_states {
            for j in 0..self.num_states {
                let mut numerator = LOGZERO;
                let mut denominator = LOGZERO;

                // for each i->j transition, calculate numerator and denominator (normalizer)
                for t in 0..(self.len_obs-1) {
                    numerator = eln_sum(
                        numerator,
                        self.log_xi[[i, j, t]]
                    );
                    for k in 0..self.num_states {
                        denominator = eln_sum(
                            denominator,
                            self.log_xi[[i, k, t]]
                        );
                    };
                };
                
                // normalize numerator with denominator and assign its exponent to a_hat[i, j]
                self.trans_mat[[i, j]] = eexpo(
                    eln_product(
                        numerator,
                        -denominator
                    )
                );
            };
        };
    }

    // update emit_mat estimate
    fn _update_emit_mat(&mut self) {
        // iterate over each emissions from state j to observation k
        for k in 0..self.num_obs {
            for j in 0..self.num_states {
                let mut numerator = LOGZERO;
                let mut denominator = LOGZERO;

                for t in 0..self.len_obs {

                    // sum observations from state j where value k was observed
                    if (self.observations[t] )== k {
                        numerator = eln_sum(
                            numerator,
                            self.log_gamma[[j, t]]
                        );
                    };

                    // sum over all observation from state j
                    denominator = eln_sum(
                        denominator,
                        self.log_gamma[[j, t]]
                    );
                };

                // estimate the prob of emitting k from state j
                self.emit_mat[[j, k]] = eexpo(
                    eln_product(
                        numerator,
                        -denominator
                    )
                );

            }
            
        };
    }

    // compute viterbi matrix and backpointer matrix
    fn _viterbi(&mut self) {
        // initialize
        for ind_state in 0..self.num_states {
            self.log_viterbi[[ind_state, 0]] = eln_product(
                eln(self.init_dist[ind_state]), 
                eln(self.emit_mat[[ind_state, self.observations[0]]])
            );
            self.backpointer_mat[[ind_state, 0]] = 0;
        };

        for ind_obs in 1..self.len_obs {                     // for each observation along the observations sequence
            for ind_curr_state in 0..self.num_states {       // for each state

                let mut vprob_temp = LOGZERO;
                let mut bp_ind_temp = 0;           
                for ind_prev_state in 0..self.num_states{
                    let v_ = eln_product(
                        self.log_viterbi[[ind_prev_state, ind_obs-1]], 
                        eln_product(
                                eln(self.trans_mat[[ind_prev_state, ind_curr_state]]),
                                eln(self.emit_mat[[ind_curr_state, self.observations[ind_obs]]])
                        )
                    );

                    if (v_ > vprob_temp) || (vprob_temp.is_nan()) {
                        vprob_temp = v_;
                        bp_ind_temp = ind_prev_state;
                    }

                }

                self.log_viterbi[[ind_curr_state, ind_obs]] = vprob_temp;
                self.backpointer_mat[[ind_curr_state, ind_obs]] = bp_ind_temp;
            
            }
        }

        let mut best_path_prob = LOGZERO;
        for (i, row) in self.log_viterbi.axis_iter(Axis(0)).enumerate() {
            let prob_temp = row[self.len_obs - 1];
            if  (prob_temp > best_path_prob) || (best_path_prob.is_nan()) {
                best_path_prob = prob_temp;
                self.best_backpointer = i;
            }
        }
    }

    /// Traceback the backpointers on the backpointer matrix from Viterbi algorithm
    fn _traceback_viterbi(&mut self) {
        self.best_state_sequence.clear();

        let mut i = self.len_obs - 1;
        let mut j = self.best_backpointer;
        self.best_state_sequence.push(self.best_backpointer);

        while i > 0 {
            let prev_state = self.backpointer_mat[[j, i]];
            self.best_state_sequence.push(prev_state);
            j = prev_state;
            i -= 1;
        }

        self.best_state_sequence.reverse();

    }

}


/// Calculate viterbi probability
///
/// INPUTS:
/// obs: sequence of observations represented as indices in emit_mat
/// init_dist: initial distribuition of states. it assumes its index corresponds to row index of trans_mat and emit_mat (for the former, both row & column indices)
/// trans_mat: state transition matrix
/// emit_mat: emission matrix. sates along the row, possible observations along the column
pub fn viterbi(obs:&Vec<usize>, init_dist: &Vec<f64>, trans_mat: &Array2<f64>, emit_mat: &Array2<f64>) -> (Array2<f64>, Array2<usize>) /*f64*/ {
    let len_obs = obs.len();                    // length of observations
    let num_states = trans_mat.shape()[0];      // number of possible states
    let mut viterbi_mat = Array2::<f64>::zeros((num_states, len_obs));  // viterbi matrix
    let mut backpointer_mat = Array2::<usize>::ones((num_states, len_obs));  // backpointer matrix

    // if the shape doesn't match, raise exception
    if emit_mat.shape()[0] != num_states {
        panic!{"Number of rows in emit_mat does not match with that of trans_mat."}
    };

    // if sume along the row for emit_mat does not equal 1 { panic! }

    // initialize
    for ind_state in 0..num_states {
        viterbi_mat[[ind_state, 0]] = init_dist[ind_state] * emit_mat[[ind_state, obs[0]]];
        backpointer_mat[[ind_state, 0]] = 0;
    };


    for ind_obs in 1..len_obs {                     // for each observation along the observations sequence
        for ind_curr_state in 0..num_states {       // for each state
            // calculate the probability of seeing the observation obs[ind_obs] for state ind_current_state as the max probabilities
            // of coming from all potential paths.
            let mut vprob_temp = 0.0;
            let mut bp_ind_temp = 0;           
            for ind_prev_state in 0..num_states{
                let v_ = viterbi_mat[[ind_prev_state, ind_obs-1]] * trans_mat[[ind_prev_state, ind_curr_state]] * emit_mat[[ind_curr_state, obs[ind_obs]]];
                if v_ > vprob_temp {
                    vprob_temp = v_;
                    bp_ind_temp = ind_prev_state;
                }
            }

            viterbi_mat[[ind_curr_state, ind_obs]] = vprob_temp;
            backpointer_mat[[ind_curr_state, ind_obs]] = bp_ind_temp;
        }
    }

    let mut best_path_prob = 0.0;
    // let mut best_bp = 0;
    for row in viterbi_mat.axis_iter(Axis(0)) {//.enumerate() {
        let prob_temp = row[len_obs - 1];
        if  prob_temp > best_path_prob {
            best_path_prob = prob_temp;
            // best_bp = i;
        }
    }

    (viterbi_mat, backpointer_mat)
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


