# rusty_hmm
Quick implementation of Hidden Markov Model in Rust.

Used the following as references:
- https://web.stanford.edu/~jurafsky/slp3/A.pdf
- http://bozeman.genome.washington.edu/compbio/mbt599_2006/hmm_scaling_revised.pdf


On HMM struct implemented here:
- HMM struct takes in observations and initial estimates of the initial state distribution, transition matrix, and emission matrix.
- Its train() function recursively updates the model
- Best estimate of the hidden state sequence can be accessed after training via .best_state_sequence attribute.


* This was mainly written for my own learning project and there are likely errors in some places.
