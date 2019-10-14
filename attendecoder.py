# encoding = utf-8 


import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope

from decoder import Decoder
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear

class AttnDecoder(Decoder):
    def __init__(self, isTraining, **attr, hp):
        '''
        Init class from base class
        '''
        super(AttnDecoder, self).__init__(isTraining, **attr, hp)

    def decode(self, decoder_inp, seq_len, encoder_hidden_states,
    	       final_state, seq_len_inp):
    	'''
    	attention-based decoder using LSTM+attention to model output seq
    	1: prepare the decoder input
    	2: emded the input
    	3: obtain the relevant loop function
    	'''
    	decoder_inputs, loop_funtion = self.prepare_decoder_input(decoder_inp)

    	#TensorArray is used to do dynamic looping over decoder input
    	inputs_ta = tf.TensorArray(size = self.max_output, dtype = tf.float32)
        inputs_ta = inputs_ta.unstack(decoder_inputs)
        
        batch_size = tf.shape(decoder_inputs)[1]
        embedding_size = decoder_inputs.get_shape()[2].value

        with variable_scope.variable_scope('attention_based'):
            attn_length = tf.shape(encoder_hidden_states)[1]
            attn_size = encoder_hidden_states.get_shape()[2].value

            #To calculate W1 * h_t, we use a 1-by-1 convolution, need to reshape before
            hidden = tf.expand_dims(encoder_hidden_states, 2)

            attention_vec_size = 64

            k = variable_scope.get_variable('AttnW', [1, 1, attn_size, attention_vec_size])
            hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], 'SAME')
            v = variable_scope.get_variable('AttnV', [attention_vec_size])

            batch_attn_size = array_ops.stack([batch_size, attn_size])
            attn = array_ops.zeros(batch_attn_size, dtype = tf.float32)
            attn.set_shape([None, attn_size])

            batch_alpha_size = array_ops.stack([batch_size, attn_length, 1, 1])
            alpha = array_ops.zeros(batch_alpha_size, dtype = tf.float32)

            attn_mask = tf.sequence_mask(tf.cast(seq_len_inp, tf.int32), dtype = tf.float32)

            def attn_loop_function(time, cell_output, state, loop_state):
                def attention(query, prev_alpha):
                    with variable_scope.variable_scope('Attention'):
                        y = linear(query, attention_vec_size, True)
                        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                        s = math_ops.reduce_sum(v * math_ops.tanh(hidden_features + y), [2, 3])
                        alpha = nn_ops.softmax(s) * attn_mask
                        sum_vec = tf.reduce_sum(alpha, reduction_indices = [1], keep_dims = True) + 1e-12
                        norm_term = tf.title(sum_vec, tf.stack([1, tf.shape(alpha)[1]]))
                        alpha = alpha / norm_term
                        alpha = tf.expand_dims(alpha, 2)
                        alpha = tf.expand_dims(alpha, 3)

                        #Now calculate the attention-weighted vector d.
                        d = math_ops.reduce_sum(alpha * hidden, [1, 2])
                        d = array_ops.reshape(d, [-1, attn_size])

                    return tuple([d, alpha])    

