

import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell


class Decoder(object):
    '''
    base class for decoder
    '''
    def __init__(self, isTraining, hp):
        '''
        init
        '''
        self.isTraining = isTraining
        if self.isTraining:
            self.out_prob = hp.decoder_out_prob
            self.isSampling = False
            if hp.decoder_samp_prob > 0.0:
                self.isSampling = True
                self.samp_prob = hp.decoder_samp_prob

        self.hidden_size = hp.decoder_hidden_size
        self.num_layers = hp.decoder_num_layers
        self.vocab_size = hp.num_decoder_symbols
        self.cell = self.set_cell_config()        
        self.emb_size = hp.decoder_embedding_size
        self.max_output = 100 #max length of output
        if self.max_output != hp.decoder_max_output:
            self.max_output = hp.decoder_max_output	


    def set_cell_config(self):
    '''
    '''
    cell = rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
    if self.isTraining:
        cell = rnn_cell.DropoutWrapper(cell, output_keep_prob = self.out_prob)

    if self.num_layers > 1:
        #if rnn is stacked then we use MultiRNNCell class
        cell = rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple = True)

    cell = rnn_cell.OutputProjectionWrapper(cell, self.vocab_size)
    return cell


    def prepare_decoder_input(self, decoder_inputs):
        '''
        input:
            decoder_inputs: the major decoder IDs
        return:
            embedded_inp: embedded decoder input
            loop_functions: function for getting next timestamp input                
        '''
        with variable_scope.variable_scope('decoder'):
            # create an embedding matrix
            embedding = variable_scope.get_variable('embedding',
                                                    [self.vocab_size, self.emb_size],
                                                    initializer = tf.random_uniform_inializer(-1.0, 1.0))
            #embed the decoder input via embedding loopup operation
            embedding_inp = embedding_ops.embedding_loopup(embedding, decoder_inputs)

        if self.isTraining:
            if self.isSampling:
                # this loop function samples the output from posterior
                # and embeds this output
                loop_function = self._sample_argmax(embedding)
            else:
                loop_function = None    
        else:
            # get loop function that would embed the maximum posterior symbol.
            # this function is used during decoding in RNNs
            loop_function = self._get_argmax(embedding)
        return (embedded_inp, loop_function)

    @abstractmethod
    def decode(self, decoder_inp, seq_len,
               encoder_hidden_states, final_state, seq_len_inp):
    '''
    input:
        decodr_inp:decoder IDs T*B, contains groudn truth during training,test time are dummy value
        seq_len:output sequence length for each input in minibatch
        encoder_hidden_states:batch major output, shape B*T*H
        final_state: Final hidden state of encoder RNN, useful for initializing decoder RNN
        seq_len_inp: useful with attention-enabled decoders to mask outputs corresponding to padding symbols
    returns:
        outputs:Time major output, T*B*|V|, of decoder RNN

    '''
    
    def _get_argmax(self, embedding):
        '''
        input:
            embedding: embedding maxtrix for embedding the symbol
        output:
            loop_function:a function returns the embedded output symbol with 
            max prob(logit score)
        '''
        def loop_function(logits):
            max_symb = math_ops.argmax(logits, 1)
            emb_symb = embedding_ops.embedding_loopup(embedding, max_symb)
            return emb_symb

        def _sample_argmax(self, embedding):
        '''
        input:
            embedding: embedding maxtrix for embedding the symbol
        return:
            loop_function: a function that samples the output symbol from
            posterior and embeds the sampled symbol
        '''
        def loop_function(prev):
        '''
        input:
            prev: logit score for previous step output
        returns:
            emb_prev: the embedding of output symbol sampled from
                    posterior over previous output    
        '''
        #tf.multinomial performs sampling given the logit scores
        # Reshaping is required to remove the extra dimension introduced
        # by sampling for a batch size of 1
        prev_symbol = tf.reshape(tf.multinomial(prev, 1), [-1])
        emb_prev = embedding_ops.embedding_loopup(embedding, prev_symbol)
        return emb_prev    