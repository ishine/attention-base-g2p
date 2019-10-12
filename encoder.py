

import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell

class Encoder(object):
    '''
    Encoder class
    '''	
    def __init__(self, isTraining, **attr,):
        '''
        init Encoder
        '''
        self.isTraining = isTraining
        if self.isTraining:
            self.out_prob = attr['out_prob']
        self.hidden_size = hp.encoder_hidden_size
        self.num_layers = hp.encoder_num_layers
        self.bi_dir = hp.encoder_bi_dir

        #hidden unit use LSTM cell
        self.cell = rnn_cell.BasicLSTMCell(self.hidden_size, stable_is_tuple = True)
        if self.isTraining:
            self.cell = run_cell.DropoutWrapper(self.cell, output_keep_prob = self.out_prob)
        self.vocab_size = hp.num_encoder_symbols
        self.emb_size = hp.embedding_size

    def _layer_encoder_input(self, encoder_inputs, seq_len, layer_depth = 1):
        '''
        inputs:
            encoder_inputs: 3-d Tensor shape T*B*E;T:number of timesteps,
                            B:batch size
                            E:input demension at every timestep
            seq_len: 1-d sequence length of each input
            layer_depth: the current layer index
        return:
            encoder_outputs:output LSTM, 3-d Tensor, shape = T*B*H 
            final state: Final hidden state of LSTM                   
        '''
        with variable_scope.variable_scope('RNNLayer%d' % (layer_depth), 
        	                               initializer = tf.random_uniform_initializer(-0.075, 0.075)):
            if self.bi_dir:
                (encoder_output_fw, encoder_output_bw), (final_state_fw, _) = \
                 rnn_bidirectional_dynamic_rnn(self.cell, self.cell,
                 	                          encoder_inputs,
                 	                          sequence_length = seq_len,
                 	                          dtype = tf.float32,
                 	                          time_major = True)
                #concatenate the output of forward and backward layer
                encoder_outputs = tf.concat([encoder_output_fw, encoder_output_bw], 2)
                #assume the final state is simply the final state of forward layer
                #a combination of hidden states can also be done
                final_state = final_state_fw
            else:
                encoder_outputs, final_state = rnn.dynamic_rnn(self.cell, encoder_inputs,
                	                                          sequence_length = seq_len, dtype = tf.float32,
                	                                          time_major = True) 
            return encoder_outputs, final_state    	                                              	                                    


    def encode_input(self, encoder_inp, seq_len):
        '''
        inputs
            encoder_inputs pass through embedding layes before feed LSTM layers
            seq_len is the input time length of the sequence
        return:
            attention states of every input timestep
            final state: final state of the LSTM
        '''
        with variable_scope.variable_scope('encoder'):
            embedding = variable_scope.get_variable('embedding', 
                                                    [self.vocab_size, self.embedding_size],
                                                    initializer = tf.random_uniform_initializer(-1.0, 1.0))
            #input ids are first embedded via embedding lookup operation
            encoder_inputs = embedding_ops.embedding_lookup(embedding, encoder_inp)
            final_states = []
            for layer_depth in xrange(self.num_layers):
                encoder_outputs, layer_final_state = self._layer_encoder_input(encoder_inputs, seq_len, layer_depth)

                #