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
    with tf.Session(config = tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads = 2)) as sess:
        with tf.variable_scope('model', reuse = None):
            model, steps_done = get_model(sess, isTraining = True)
        with tf.variable_scope("mode", reuse = True):
            mvalid = create_model_graph(sess, isTraining = False)        
    	
        print('Models created')
        print('Reading data from %s' % hp.data.dir)
        sys.stdout.flush()

        # Load train and dev data
        train_data = data_utils.read_and_bucket_data(os.path.join(hp.data_dir, 'train.pkl'))
        dev_set = data.utils.read_and_bucket_data(os.path.join(hp.data.dir, 'dev.pkl'))
        
        step_time, loss = 0.0, 0.0
        previous_losses = []
        epoch_id = model.epoch.eval()

        val_wer_window = []
        window_size = 3
        if steps_done > 0:
            # The model saved would have wer and per better than 1.0
            best_wer, _ = calc_levenshtein_loss(mvalid, sess, dev_set)
        else:
            best_wer = 1.0

        print('----- Start training!! -----\n')
        sys.stdout.flush()
        while (epoch_id < hp_max_epochs):
            steps = 0.0
            # Batch the data (Also shuffles the data)
            batch_data = data.utils.batch_bucketed_data(train_data, batch_size = hp.batch_size)
            for batch in batch_data:
                # Run a minibatch update and record the run times
                start_time = time.time()
                encoder_inputs, seq_len, decoder_inputs, seq_len_target = model.get_batch(batch)
                _, step_loss = model.step(sess, encoder_inputs, seq_len, decoder_inputs, seq_len_target)
                
                step_time += (time.time() - start_time)
                loss += step_loss

                steps += 1.0

            # increase the epoch counter
            epoch_id += 1
            sess.run(model.epoch_incr)
            step_time /= steps
            loss /= steps
            perplexity = np.exp(loss) if loss < 300 else float('inf')
            print('Epoch %d global step %d learning rate %.4f step-time %.2f'
                  'perplexity %.4f' % (epoch_id, model.global_step.eval(),
                                       model.lr.eval(), step_time,
                                       perplexity) )
            if len(previuos_losses) >= 3 and loss > max(previous_lossess[-3:]):
                sess.run(model.lr_decay)
            previous_losses.append(loss)
            step_time, loss = 0.0, 0.0

            #Calculate validation result
            val_wer, val_per = calc_levenshtein_loss(mvalid, sess, dev_set)
            print('validation WER: %.5f, PER: %.5f' % (val_wer, val_per))
            sys.stdout.flush()

            #Validation WER is a moving window, we add the new entry and pop the oldest one
            val_wer_window.append(val_wer)
            if len(val_wer_window) > window_size:
                val_wer_window.pop(0)
            avg_wer = sum(val_wer_window) / float(len(val_wer_window))
            print('Average validation WER %.5f' % (avg_wer))
            sys.stdout.flush()

            # the best model is decided based on average validation WER to
            # remove noisy cases of one off validation success
            if best_wer > avg_wer:
                # Save the best model
                best_wer = avg_wer
                print('Saving Updated Model')
                sys.stdout.flush()
                checkpoint_path = os.path.join(hp.train_dir, 'g2p.ckpt')
                model.saver.save(sess, checkpoint_path,
                                 global_step = model.global_step,
                                 write_meta_graph = False)
  

def calc_levenshtein_loss(model, sess, eval_set):
    ''' 
    calculate the actual loss funtion
    input:
        model: seq2seq modle instance
        sess: Tensorflow sess with model computation graph
        eval_set: Bucketed evaluation set
    returns:
        wer: Word Error rate
        per: Phone Error Rate    
    '''
    total_words = 0
    total_phonemes = 0
    wer = 0
    per = 0
    edit_distances = []
    for bucket_id in xrange(len(data_utils._buckets)):
        cur_data = eval_set[buckit_id]
        for batch_offset in xrange(0, len(cur_data), hp.batch_size):
            batch = cur_data[batch_offset : batch_offset + hp.batch_size]
            num_instances = len(batch)
            #Each instance is ap pair of ([Input sequence], [Output sequence])
            inp_ids = [inst[0] for inst in batch]
            gt_ids = [inst[1] for inst in batch]
            encoder_inputs, seq_len, decoder_inputs, seq_len_target = model.get_batch(batch, buckit_id = buckit_id)
            # Run the model to get output_logits of shape T*B*|v|
            output_logits = model.step(sess)

if __name__ == '__main__':
    main()	