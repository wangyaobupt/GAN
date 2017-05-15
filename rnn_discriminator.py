import tensorflow as tf
import numpy as np

def discriminator(inputTensor,reuseCell):
    with tf.name_scope('D_net'):
        num_units_in_LSTMCell = 1 
        lstmCell = tf.contrib.rnn.BasicLSTMCell(num_units_in_LSTMCell, reuse=reuseCell)
        init_state = lstmCell.zero_state(100, dtype=tf.float32)
        raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, inputTensor, initial_state=init_state)
        output_logits = tf.unstack(tf.transpose(raw_output, [1, 0, 2]), name='outList')
        d_logit = output_logits[-1];
        d_logit = tf.identity(d_logit, 'd_net_logit')
        return d_logit

def genData(n_samples, len_data):
    result = np.ones((n_samples, len_data, 1))
    step_vector = (1 + 10*np.random.random([n_samples,1]))
    step_vector = step_vector.astype(int)
    for seqIdx in range(1, len_data):
        for sampleIdx in range(n_samples):
            result[sampleIdx][seqIdx][0] = result[sampleIdx][seqIdx-1][0] + step_vector[sampleIdx][0]
    return result

if __name__ == '__main__':
    d_inputTensor = tf.placeholder(tf.float32, shape=(None, 10, 1))
    label_tensor = tf.placeholder(tf.float32, shape=(None, 1))

    d_logit = discriminator(d_inputTensor, None)

    loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_logit,
                    labels=label_tensor
                    ),
                name='d_loss'
                )
    tf.summary.scalar('d_loss',loss)
    
    train_d = tf.train.AdamOptimizer(0.001).minimize(loss)

    # visualize and model saving
    merged = tf.summary.merge_all()
    all_vars = tf.global_variables()
    saver = tf.train.Saver(all_vars)
    sess = tf.Session()
    tf_writer = tf.summary.FileWriter('tf_writer', sess.graph)
    sess.run(tf.global_variables_initializer())
    
    # generate data
    iData_Positive = genData(1000, 10)
    iData_Negative = np.random.uniform(0,10,size=[1000,10, 1])
    iData = np.concatenate((iData_Positive, iData_Negative))
    iLabel = np.concatenate((np.ones((1000,1)), np.zeros((1000,1))))

    batch_size = 100
    numIteration = 1000
    n_samples = iData.shape[0]
    n_batches = n_samples / batch_size
    for iterIdx in range(numIteration):
        for batchIdx in range(n_batches):
            data_batch = iData[batchIdx*batch_size:(batchIdx+1)*batch_size]
            label_batch =  iLabel[batchIdx*batch_size:(batchIdx+1)*batch_size]
            
            sess.run(train_d, 
                feed_dict={
                    d_inputTensor:data_batch, 
                    label_tensor:label_batch})


            summary = sess.run(merged,
                feed_dict={
                    d_inputTensor:data_batch, 
                    label_tensor:label_batch})

            d_log = sess.run(d_logit,
                feed_dict={
                    d_inputTensor:data_batch, 
                    label_tensor:label_batch})
        tf_writer.add_summary(summary, iterIdx)
        tf_writer.flush()
        if iterIdx%100 == 0:
            print 'IterIdx=', iterIdx, ' d_logit[0]=', d_log[0]
            print 'data_batch[0] = ', data_batch[0],'label_batch[0]=', label_batch[0]

