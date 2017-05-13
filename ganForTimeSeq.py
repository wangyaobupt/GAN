import numpy as np
import tensorflow as tf

class GANForTimeSeq:
    log_path = 'tf_writer'

    def __init__(self, lenOfTimeSeq, lr_g=0.01, lr_d=0.01, useGPU=True):
        self.useGPU = useGPU
        self.seq_len = lenOfTimeSeq
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.n_classes = 2

        # define common parameter
        self.batch_size_t = tf.placeholder(tf.int32, None)
        
        # g_network data flow
        g_inputTensor = tf.random_uniform([self.batch_size_t, self.seq_len], minval=0, maxval=10)
        g_logit = self.generator(g_inputTensor)
        
        # d-network data flow
        self.groundTruthTensor = tf.placeholder(tf.float32,shape=(None, self.seq_len, 1),name='gndTruth')
        self.d_logit_gnd_truth = self.discriminator(self.groundTruthTensor, None)
        self.g_logit = tf.reshape(g_logit, [-1, self.seq_len, 1])
        self.d_logit_fake = self.discriminator(self.g_logit, True)
        
        # define loss function
        with tf.name_scope('Loss'):
            # For G-network, the more generated data is judged as "TRUE", the less the loss would be
            g_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.genOneHotVector(0),
                    logits=self.d_logit_fake),
                name='g_loss'
                )
            tf.summary.scalar('g_loss',g_loss)
            
            # For D-network, jduge ground truth to TRUE, jduge G-network output to FALSE,making loss low
            d_loss_ground_truth = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.genOneHotVector(0),
                    logits=self.d_logit_gnd_truth),
                name='d_loss_gnd'
                )
            tf.summary.scalar('d_loss_gnd_truth',d_loss_ground_truth)
            d_loss_fake = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.genOneHotVector(1),
                    logits=self.d_logit_fake),
                name='d_loss_fake'
                )
            tf.summary.scalar('d_loss_fake',d_loss_fake)
            d_loss = d_loss_ground_truth + d_loss_fake
            tf.summary.scalar('d_loss',d_loss)

        with tf.name_scope('Accuracy'):
            correct_pred_gnd_truth = tf.equal(tf.argmax(self.d_logit_gnd_truth, 1), tf.argmax(self.genOneHotVector(0), 1))
            d_accuracy_gnd_truth = tf.reduce_mean(tf.cast(correct_pred_gnd_truth, tf.float32))
            tf.summary.scalar('d_acc_gnd_truth',d_accuracy_gnd_truth)
            
            correct_pred_fake = tf.equal(tf.argmax(self.d_logit_fake, 1), tf.argmax(self.genOneHotVector(1), 1))
            d_accuracy_fake = tf.reduce_mean(tf.cast(correct_pred_fake, tf.float32))
            tf.summary.scalar('d_acc_fake', d_accuracy_fake)

        # Optimize ops
        self.train_g = tf.train.AdamOptimizer(self.lr_g).minimize(g_loss)
        self.train_d = tf.train.AdamOptimizer(self.lr_d).minimize(d_loss)

        # visualize and model saving
        self.merged = tf.summary.merge_all()
        all_vars = tf.global_variables()
        saver = tf.train.Saver(all_vars)
        if self.useGPU:
            self.sess = tf.Session()
        else:
            print 'Not use GPU'
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
        self.tf_writer = tf.summary.FileWriter(GANForTimeSeq.log_path, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    # batch_size: how many sample trained inone batch
    # numIteration: number of interation loops
    # gnd_ruth_tensor: ground truth tensor, in shape of (n_samples, seq_len, 1)
    def train(self, batch_size, numIteration, gnd_truth_tensor):
        n_samples = gnd_truth_tensor.shape[0]
        n_batches = n_samples / batch_size
        print 'n_samples=',n_samples, ' ,n_batches=',n_batches
        for iterIdx in range(numIteration):
            for batchIdx in range(n_batches):
                gnd_truth_batch = gnd_truth_tensor[batchIdx*batch_size:(batchIdx+1)*batch_size]
                
                for index in range(2):
                    self.sess.run(self.train_g, 
                            feed_dict={self.batch_size_t:batch_size, self.groundTruthTensor:gnd_truth_batch})

                for index in range(3):
                    self.sess.run(self.train_d, 
                        feed_dict={self.batch_size_t:batch_size, self.groundTruthTensor:gnd_truth_batch})

                summary, g_logit = self.sess.run([self.merged,self.g_logit], 
                        feed_dict={self.batch_size_t:batch_size, self.groundTruthTensor:gnd_truth_batch})

            self.tf_writer.add_summary(summary, iterIdx)
            self.tf_writer.flush()
            if iterIdx % 100 == 0:
                print 'IterIdx=', iterIdx, ' g_logit[0]=', g_logit[0]


    # generative network
    # use multi-layer percepton to generate time sequence from random noise
    # input tensor must be in shape of (batch_size, self.seq_len)
    def generator(self, inputTensor):
        with tf.name_scope('G_net'):
            numberOfInputDims = self.seq_len
            gInputTensor = tf.identity(inputTensor, name='input')
            activation_fc1 = self.fullConnectedLayer(gInputTensor, self.seq_len, 1);
            tf.summary.histogram('activation_fc1', activation_fc1)
            activation_fc2 = self.fullConnectedLayer(activation_fc1, self.seq_len, 2);
            tf.summary.histogram('activation_fc2', activation_fc2)
            g_logit = activation_fc2
            g_logit = tf.identity(g_logit, 'g_logit')
            return g_logit

    # discriminate network#
    # Use LSTM to judge whethwer the input tensor is "ground truth"(1,0) or "generated by G-net"(0,1)
    # inputTensor must be in shape of (batch_size, seq_len, 1)
    def discriminator(self, inputTensor,reuseCell):
        with tf.name_scope('D_net'):
            num_units_in_LSTMCell = self.n_classes
            lstmCell = tf.contrib.rnn.BasicLSTMCell(num_units_in_LSTMCell, reuse=reuseCell)
            init_state = lstmCell.zero_state(self.batch_size_t, dtype=tf.float32)
            raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, inputTensor, initial_state=init_state)
            output_logits = tf.unstack(tf.transpose(raw_output, [1, 0, 2]), name='outList')
            d_logit = output_logits[-1];
            d_logit = tf.identity(d_logit, 'd_net_logit')
            return d_logit
   
    def fullConnectedLayer(self, inputTensor, numOfNodesInLayer, index):
        layerIdxStr = 'fc'+ str(index)
        numberOfInputDims = inputTensor.shape[1].value
        w = tf.Variable(initial_value=tf.random_normal([numberOfInputDims, numOfNodesInLayer]), name=('w_'+layerIdxStr))
        b = tf.Variable(tf.zeros([1, numOfNodesInLayer]), name='b_'+layerIdxStr)
        a = tf.nn.sigmoid(tf.matmul(inputTensor, w) + b, name='a_'+layerIdxStr)
        return a

    def genOneHotVector(self, class_idx):
        indices = tf.ones([self.batch_size_t],dtype=tf.int32)*class_idx
        result = tf.one_hot(indices, depth=self.n_classes, on_value = 1.0, off_value = 0.0)
        return result;
