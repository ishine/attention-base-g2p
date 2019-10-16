



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

        #then decode
        self.outputs = self.decoder.decode(self.decoder_inputs, self.seq_len_target, 
        	                               self.encoder_hidden_states, self.final_state,
        	                               self.seq_len)
        #Training outputs and losses
        self.losses = self.seq2seq_loss(self.outputs, self.targets, self.seq_len_target)

        if isTraining:
            #Gradients and parameters updation for training the model
            params = tf.trainable_variables()
            print('\nModel parameters:\n')
            for var in params:
                print(('{0}: {1}').format(var.name, var.get_shape()))
            print
            #Initialize optimizer
            opt = tr.train.AdamOptimizer(self.lr)
            #get gradient from loss
            gradients = tf.gradients(self.losses, params)
            # Clip the gradients to avoid the problem of gradient explosion
            # possible early in training
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

            self.gradient_norm = norm
            # Apply gradients
            self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step = self.global_step)

        #Model saver function
        self.saver = tf.train.Saver(tf.global_variable(), max_to_keep = 2)

    
    @staticmethod
    def seq2seq_loss(logits, targets, seq_len_target):
        '''
        input:
            logits:a 2-d tensor of shape (T*B)*|V| containing the logit score
            per output symbol
            targets: 2-d tensor of shape T*B that contains the ground truth output symbols
            seq_len_target:seq len of output seq.Required to mask padding symbols in output seq            
        '''
        with ops.name_scope('sequence_loss', [logits, targets]):
            flat_targets = tf.reshape(targets, [-1])
            cost = nn_ops.sparse_softmax_cross_entroy_with_logits(logits = logits, labels = flat_targets)

        # Mask this cost since the output seq is padded
        batch_major_mask = tf.sequence_mask(seq_len_target, dtype = tf.float32)
        time_major_mask = tf.transpose(batch_major_mask, [1, 0])
        weights = tf.reshape(time_major_mask, [-1])
        mask_cost = weights * cost

        loss = tf.reshape(mask_cost, tf.shape(targets))
        # Average the loss for each example by the # of timesteps
        cost_per_example = tf.reduce_sum(loss, reduction_indices = 0) / tf.cast(seq_len_target, tf.float32)
        # Return the average cost over all examples
        return tf.reduce_mean(cost_per_example)

    def step(self, sess, encoder_inputs, seq_len, decoder_inputs, seq_len_target):
        '''
        perform 1 minibatch update/evalution
        input:
            sess:
            encoder_inputs: list of a minibatch of input IDs
            seq_len: input seq len
            decoder_inputs: list of minibatch of output ids
            seq_len_target: output seq length
        returns:
            output of a minibatch. the exact output depends on whether the model is in
            training mode or evalution mode or evalution mode    
        '''
        # pass input via feed dict method
        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.decoder_inputs.name] = decoder_inputs
        input_feed[self.seq_len.name] = seq_len
        input_feed[self.seq_len_target.name] = seq_len_target

        if self.isTraining:
            # Important to have gradient updates as this operation is what
            # actually updates the parameters
            output_feed = [self.updates, self.gradient_norms, self.losses]
        else:
            # Evaluation    
        
        outputs = sess.run(output_feed, input_feed)
        if self.isTraining:
            return outputs[1], outputs[2]
        else:
            return outputs[0]
                    
    def get_batch(self, data, bucket_id = None):
        '''
        prepare data from given data.
        input:
            data: a list of datapoints
            bucket_id: buckit ID of data. This is irrevelant for training but for
                   evaluation we can limit the padding by the bucket size
        Returns:
            batched in IDs, input seq length, output IDs & output seq length               
    '''
    if not self.isTraining:
        _, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []
    batch_size = len(data)

    seq_len = np.zeros((batch_size), dtype = np.int64)
    seq_len_target = np.zeros((batch_size), dtype = np.int64)

    for i, sample in enumerate(data):
        encoder_input, decoder_input = sample
        seq_len[i] = len(encoder_input)
        if not self.isTraining:
            seq_len_target[i] = decoder_size
        else:
            # 1 is added to output seq length because the EOS token is
            # crucial to 'halt' the decoder. Consider it the punctuation 
            # mark of an english sentence. Both are necessary
            seq_len_target[i] = len(decoder_input) + 1

        # Max input and output length whick limit the padding till them
        max_len_source = max(seq_len)
        max_len_target = max(seq_len_target)

        for i, sample in enumerate(data):
            encoder_input, decoder_input = sample
            # Encoder inputs are padded and then reversed
            encoder_pad_size = max_len_soruce - len(encoder_input)
            encoder_pad = [data_utils.PAD_ID] * encoder_pad_size
            # Encoder input is reversed
            encoder_inputs.append(list(reversed(encoder_input)) + encoder_pad)
            
            # 1 is added to decoder_input decause GO_ID is considered a part of
            # decoder input. while EOS_ID is also added, it's really used by 
            # the target tensor (self.tensor) in the core code above
            decoder_pad_size = max_len_target - (len(decoder_input) + 1)
            decoder_inputs.append([data_utils.GO_ID] +
            	                  decoder_input +
            	                  [data_utils.EOS_ID] +
            	                  [data_utils.PAD_ID] * decoder_pad_size)
        # Both the id sequence are made time major via transpos
        encoder_inputs = np.asarray(encoder_inputs, dtype = np.int32).T
        decoder_inputs = np.asarray(decoder_inputs, dtype = np.int32).T

        return encoder_inputs, seq_len, decoder_inputs, seq_len_target