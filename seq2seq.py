



import tensorflow as tf
import numpy as np



class Seq2SeqModel(object):
    def __init__(self, buckets, isTraining, max_gradient_norm, batch_size,
    	         lr, lr_decay, encoder_attr, decoder_attr):
        '''
        '''
        self.buckets = buckets
        self.isTraining = isTraining
        self.batch_size = batch_size
        self.lr = tf.Variable(float(lr), trainable = False)
        self.lr_decay_op = self.lr.assign(self.lr, lr_decay)

        #Number of gradient updates performed
        self.global_step = tf.Variable(0, trainable = False)
        #Number of epoch done
        self.epoch = tf.Variable(0, trainable = False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)

        #Placeholder for encoder
        self.encoder_inputs = tf.placeholder(tf.int32, shape = [None, None], name = 'encoder')
        _batch_size = self.encoder_inputs.get_shape()[1].value

        #input seq length placeholder
        self.seq_len = tf.placeholder(tf.int32, shape = [_batch_size], name = 'seq_len')
        
        #output seq length placeholder
        self.seq_len_target = tf.placeholder(tf.int32, shape = [_batch_size], name = 'seq_len_target')
        
        #input to deocder RNN, This input has an inital extra symbol - GO -
        self.decoder_inputs = tf.placeholder(tf.int32, shape = [None, None], name = 'decoder')
        
        #target are decoder inputs shift by one ignore GO symbol
        self.targets = tf.slice(self.decoder_inputs, [1, 0], [-1, -1])

        #init encoder and decoder RNNs
        self.encoder = Encoder(isTraining, **encoder_attr, hp)
        if hp.is_simple_decoder:
            print('----- simple decoder not supported ------ ')
            sys.exit(1)
        else:
            self.decoder = AttnDecoder(isTraing, **decoder_attr, hp)

        #first encode input
        self.encoder_hidden_states, self.final_state = \
            self.encoder.encoder_input(self.encoder_inputs, self.seq_len)