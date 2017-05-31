import numpy as np
import tensorflow as tf

class GANForTimeSeq:
    log_path = 'tf_writer'

    def __init__(self, lenOfTimeSeq, lr_g=0.001, lr_d=0.001, useGPU=True):
        self.useGPU = useGPU
        self.seq_len = lenOfTimeSeq
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.n_classes = 2

        # define common parameter, it is a scalar
        self.batch_size_t = tf.placeholder(tf.int32, shape=[])
        
        # g_network data flow
        self.g_inputTensor = tf.placeholder(tf.float32, shape=(None, self.seq_len))
        self.g_inputLabel = tf.placeholder(tf.float32, shape=(None, self.seq_len))
        g_logit = self.generator(self.g_inputTensor)
        tf.summary.histogram('g_net_input', self.g_inputTensor)

        # d-network data flow
        self.groundTruthTensor = tf.placeholder(tf.float32,shape=(None,self.seq_len, 1),name='gndTruth')
        self.sum_gnd_truth = tf.summary.tensor_summary('gnd_truth', self.groundTruthTensor)
        tf.summary.histogram('gnd_truth', self.groundTruthTensor)
        self.d_logit_gnd_truth = self.discriminator(self.groundTruthTensor, None)
        self.sum_d_logit_gnd = tf.summary.tensor_summary('d_logit_gnd_truth', self.d_logit_gnd_truth)
        self.g_logit = tf.reshape(g_logit, [-1, self.seq_len, 1])
        tf.summary.histogram('g_logit', self.g_logit)
        self.sum_g_logit = tf.summary.tensor_summary('g_logit', self.g_logit)
        self.d_logit_fake = self.discriminator(self.g_logit, True)
        self.sum_d_logit_fake = tf.summary.tensor_summary('d_logit_fake', self.d_logit_fake)
        
        # define loss function
        with tf.name_scope('Loss'):
            # For G-network, the more generated data is judged as "TRUE", the less the loss would be
            g_loss_d = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_logit_fake,
                    labels=tf.ones(shape=[tf.reduce_mean(self.batch_size_t),1])
                    ),
                name='g_loss_d'
                )
            tf.summary.scalar('g_loss_d',g_loss_d)
           
            # use G_label to assist train of G network
           # g_loss_label = tf.reduce_mean(
           #     tf.nn.l2_loss(
           #         g_logit - self.g_inputLabel
           #         ),
           #     name='g_loss_label'
           #     )
           # tf.summary.scalar('g_loss_label',g_loss_label)

            g_loss = g_loss_d
            tf.summary.scalar('g_loss',g_loss)

            # For D-network, jduge ground truth to TRUE, jduge G-network output to FALSE,making loss low
            d_loss_ground_truth = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_logit_gnd_truth,
                    labels=tf.ones(shape=[tf.reduce_mean(self.batch_size_t),1])
                    ),
                name='d_loss_gnd'
                )
            tf.summary.scalar('d_loss_gnd_truth',d_loss_ground_truth)
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_logit_fake,
                    labels=tf.zeros(shape=[tf.reduce_mean(self.batch_size_t),1])
                    ),
                name='d_loss_fake'
                )
            tf.summary.scalar('d_loss_fake',d_loss_fake)
            d_loss = d_loss_ground_truth + d_loss_fake
            tf.summary.scalar('d_loss',d_loss)

        with tf.name_scope('Accuracy'):
            correct_pred_gnd_truth = tf.greater(self.d_logit_gnd_truth, tf.zeros([tf.reduce_mean(self.batch_size_t), 1]))
            d_accuracy_gnd_truth = tf.reduce_mean(tf.cast(correct_pred_gnd_truth, tf.float32))
            tf.summary.scalar('d_acc_gnd_truth',d_accuracy_gnd_truth)
            
            correct_pred_fake = tf.less(self.d_logit_fake, tf.zeros([tf.reduce_mean(self.batch_size_t), 1]))
            d_accuracy_fake = tf.reduce_mean(tf.cast(correct_pred_fake, tf.float32))
            tf.summary.scalar('d_acc_fake', d_accuracy_fake)

        # Optimize ops
        g_net_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G_net')
        g_net_var_list = g_net_var_list +  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g_rnn')
        print g_net_var_list
        self.train_g = tf.train.AdamOptimizer(self.lr_g).minimize(g_loss,var_list=g_net_var_list)
        d_net_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D_net')
        d_net_var_list = d_net_var_list +  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='d_rnn')
        print d_net_var_list
        self.train_d = tf.train.AdamOptimizer(self.lr_d).minimize(d_loss,var_list=d_net_var_list)

        # visualize and model saving
        self.merged = tf.summary.merge_all()
        all_vars = tf.global_variables()
        self.saver = tf.train.Saver(all_vars)
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
    def train(self, batch_size, numIteration, gnd_truth_tensor, g_net_input_tensor, g_label_tensor):
        n_samples = gnd_truth_tensor.shape[0]
        sample_len = gnd_truth_tensor.shape[1]
        n_batches = n_samples / batch_size
        print 'n_samples=',n_samples, ' ,n_batches=',n_batches
        for iterIdx in range(numIteration):
            for batchIdx in range(n_batches):
                gnd_truth_batch = gnd_truth_tensor[batchIdx*batch_size:(batchIdx+1)*batch_size]
                if (g_net_input_tensor.any()):
                    g_net_input = g_net_input_tensor[batchIdx*batch_size:(batchIdx+1)*batch_size]
                else:
                    g_net_input = np.random.uniform(-1,1,(batch_size,sample_len))
                
                for index in range(1):
                    self.sess.run(self.train_d, 
                            feed_dict={
                                self.batch_size_t:batch_size, 
                                self.groundTruthTensor:gnd_truth_batch, 
                                self.g_inputTensor:g_net_input,
                                self.g_inputLabel:g_label_tensor})

                for index in range(1):
                    self.sess.run(self.train_g, 
                            feed_dict={
                                self.batch_size_t:batch_size, 
                                self.groundTruthTensor:gnd_truth_batch,
                                self.g_inputTensor:g_net_input,
                                self.g_inputLabel:g_label_tensor})

                summary, g_logit, d_logit_fake,d_logit_gnd_truth, sum_gnd_truth = self.sess.run(
                        [self.merged,self.g_logit,self.d_logit_fake, self.d_logit_gnd_truth, self.sum_gnd_truth], 
                        feed_dict={
                            self.batch_size_t:batch_size, 
                            self.groundTruthTensor:gnd_truth_batch,
                            self.g_inputTensor:g_net_input,
                            self.g_inputLabel:g_label_tensor})

            self.tf_writer.add_summary(summary, iterIdx)
            self.tf_writer.add_summary(sum_gnd_truth, iterIdx)
            self.tf_writer.flush()
            if iterIdx % 100 == 0:
                print 'IterIdx=', iterIdx, ' g_logit[0]=', g_logit[0]
                print 'gnd_truth[0] = ', gnd_truth_batch[0]
                print 'sigmoid(d_logit_fake[0]) = ', tf.sigmoid(d_logit_fake[0]).eval(session=self.sess),' sigmoid(d_logit_gnd_truth[0]) = ', tf.sigmoid(d_logit_gnd_truth[0]).eval(session=self.sess),
                self.save_model_checkpoint(GANForTimeSeq.log_path+'/', iterIdx)

    # generative network
    # use multi-layer percepton to generate time sequence from random noise
    # input tensor must be in shape of (batch_size, self.seq_len)
    def generator(self, inputTensor):
        with tf.name_scope('G_net'):
            gInputTensor = tf.identity(inputTensor, name='input')
            # Multilayer percepton implementation
            numNodesInEachLayer = 10
            numLayers = 2 
            
            previous_output_tensor = gInputTensor
            for layerIdx in range(numLayers):
                activation,z = self.fullConnectedLayer(previous_output_tensor, numNodesInEachLayer, layerIdx)
                previous_output_tensor = activation
                tf.summary.histogram('z_'+str(layerIdx), z)
                tf.summary.histogram('activation_fc'+str(layerIdx), activation)
            g_logit = z
            g_logit = tf.identity(g_logit, 'g_logit')
            
            ## LSTM implementation
            #seq_len = int(gInputTensor.shape[1])
            #max_time = seq_len
            #gInputTensor = tf.reshape(gInputTensor,[-1,max_time,1])
            #num_units_in_LSTMCell = 10 
            #with tf.variable_scope('g_rnn'):
            #    lstmCell = tf.contrib.rnn.BasicLSTMCell(num_units_in_LSTMCell)
            #    init_state = lstmCell.zero_state(self.batch_size_t, dtype=tf.float32)
            #    raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, gInputTensor, initial_state=init_state)
            #rnn_output_list = tf.unstack(tf.transpose(raw_output, [1, 0, 2]), name='outList')
            #rnn_output_tensor = rnn_output_list[-1];
            #g_sigmoid, g_logit = self.fullConnectedLayer(rnn_output_tensor, seq_len , 1)
            #g_logit = tf.identity(g_logit, 'g_net_logit')
            return g_logit

    # discriminate network#
    # Use LSTM to judge whethwer the input tensor is "ground truth"(1) or "generated by G-net"(0)
    # inputTensor must be in shape of (batch_size, seq_len, 1)
    def discriminator(self, inputTensor,reuseParam):
        with tf.name_scope('D_net'):
            num_units_in_LSTMCell = 10
            
            # RNN definition
            with tf.variable_scope('d_rnn'):
                lstmCell = tf.contrib.rnn.BasicLSTMCell(num_units_in_LSTMCell,reuse=reuseParam)
                init_state = lstmCell.zero_state(self.batch_size_t, dtype=tf.float32)
                raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, inputTensor, initial_state=init_state)
            
            rnn_output_list = tf.unstack(tf.transpose(raw_output, [1, 0, 2]), name='outList')
            rnn_output_tensor = rnn_output_list[-1];
            
            # Full connected network
            numberOfInputDims = inputTensor.shape[1].value
            numOfNodesInLayer = 1
            if not reuseParam:
                self.d_w = tf.Variable(initial_value=tf.random_normal([numberOfInputDims, numOfNodesInLayer]),
                        name=('dnet_w_1'))
                self.d_b = tf.Variable(tf.zeros([1, numOfNodesInLayer]), name='dnet_b_1')
            self.d_z = tf.matmul(rnn_output_tensor,self.d_w) + self.d_b
            self.d_z = tf.identity(self.d_z, name='dnet_z_1')
            d_sigmoid = tf.nn.sigmoid(self.d_z, name='dnet_a_1')

            d_logit = self.d_z
            d_logit = tf.identity(d_logit, 'd_net_logit')
            return d_logit
   
    def fullConnectedLayer(self, inputTensor, numOfNodesInLayer, index):
        layerIdxStr = 'fc' + str(index)
        numberOfInputDims = inputTensor.shape[1].value
        w = tf.Variable(initial_value=tf.random_normal([numberOfInputDims, numOfNodesInLayer]),
                        name=('w_' + layerIdxStr))
        b = tf.Variable(tf.zeros([1, numOfNodesInLayer]), name='b_' + layerIdxStr)
        z = tf.matmul(inputTensor, w) + b
        z = tf.identity(z, name='z_' + layerIdxStr)
        a = tf.nn.sigmoid(z, name='a_' + layerIdxStr)
        return a, z

    def genOneHotVector(self, class_idx):
        indices = tf.ones([tf.reduce_mean(self.batch_size_t)],dtype=tf.int32)*class_idx
        result = tf.one_hot(indices, depth=self.n_classes, on_value = 1.0, off_value = 0.0)
        return result;

    def save_model_checkpoint(self, path, step_idx):
        self.saver.save(self.sess, path + "model.ckpt", global_step=step_idx)
        print("Model saved as " + path + "model.ckpt")
