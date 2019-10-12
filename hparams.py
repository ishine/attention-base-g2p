import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
    lr = 1e-3,
    lr_decay = 0.8,
    max_gnorm = 5.0,
    batch_size = 256,
    embedding_size = 512,
    hidden_size = 512,
    num_layers = 3,
    out_prob = 0.8,


    num_check = 100,
    max_epochs = 100,
    
	)
