

import tensorflow as tf
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