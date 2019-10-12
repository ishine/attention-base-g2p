import sys
import tensorflow as tf

def create_model_graph(sess, isTraining, hp):
    '''
    get seq2seq model
    '''    	
    return seq2seq.Seq2SeqModel(_buckets, isTraining, hp.max_gradient_norm
                                hp.batch_size, hp.lr, hp.lr_decay, FLAGS.encoder_attribs,
                                FLAGS.decoder_attribs)

def get_model(session, isTraining = True):
    '''
    creat model from previous checkpoint
    '''
    model = create_model_graph()

def train(hp):
    with tf.Session(config = tf.ConfigProto(intra_op_parallelism_threads=2,
    				inter_op_parallelism_threads = 2)) as sess:
        with tf.variable_scope('model', reuse = None):
            model, steps_done = get_model(sess, isTraining = True)
        with tf.variable_scope("mode", reuse = True):
            mvalid = create_model_graph(sess, isTraining = False)        
    	





if __name__ == '__main__':
    main()	