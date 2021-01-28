import tensorflow as tf
import numpy as np
import sys, re
from models.model1 import focal_loss_sigmoid
sys.path.append('/home/chzze/bitbucket/SSC_rls')

from tensorflow.contrib import layers
import os
import json


def get_train_info(data_path, key):
    with open(os.path.join(data_path, '.info'), 'r') as f:
        info = json.load(f)
        return info[key]


def calculate_accuracy(prob, label):
    predicted = tf.cast(prob > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, label), dtype=tf.float32))
    return accuracy


def design_scope(class_name):
    model_scope = re.sub('Inference', '', class_name)
    classifier_scope = re.sub('Model', 'Classifier', model_scope)
    return model_scope, classifier_scope


def set_scope(class_name, scope_name):
    model_scope = re.sub('Inference', '', class_name)
    new_scope = re.sub('Model', scope_name, model_scope)
    return new_scope


def get_global_vars(scope_list):
    _vars = []
    for scope in scope_list:
        _vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return _vars


def get_prep_conv_vars(model_name):
    scope_list = [model_name, 'Classifier']
    _vars = get_global_vars(scope_list)
    return _vars


def feature_standardize(x):
    eps = 1e-8
    mean = tf.reduce_mean(x)
    var = tf.reduce_mean(tf.pow(x, 2)) - mean**2
    standardized_x = (x - mean) / tf.sqrt(var + eps)
    return standardized_x


def bottleneck_layer(input_x, growth_k, is_training, use_dropout=False, dropout_rate=0.2):
    out = layers.batch_norm(inputs=input_x, center=True, scale=True, is_training=is_training)
    out = tf.nn.relu(out)
    out = layers.conv2d(inputs=out, num_outputs=4*growth_k, kernel_size=1, stride=1,
                        padding='SAME', activation_fn=None)
    out = layers.batch_norm(inputs=out, center=True, scale=True, is_training=is_training)
    out = tf.nn.relu(out)
    out = layers.conv2d(inputs=out, num_outputs=growth_k, kernel_size=3, stride=1,
                        padding='SAME', activation_fn=None)
    if use_dropout:
        out = tf.layers.dropout(inputs=out, rate=dropout_rate, training=is_training)
    return out


def transition_layer(input_x, layer_name, is_training, theta=1.0, reduction_ratio=16,
                     last_layer=False):  # Model58 serial ~23 theta=0.5
    with tf.name_scope(layer_name):
        in_channel = input_x.shape[-1].value
        out = layers.batch_norm(inputs=input_x, center=True, scale=True, is_training=is_training)
        out = tf.nn.relu(out)
        out = layers.conv2d(inputs=out, num_outputs=int(in_channel*theta), kernel_size=1, stride=1,
                            padding='SAME', activation_fn=None)

        if last_layer is False:
            squeeze = tf.reduce_mean(out, axis=[1, 2], keepdims=True)  # global average pooling
            excitation = layers.fully_connected(inputs=squeeze,
                                                num_outputs=squeeze.shape[-1].value // reduction_ratio,
                                                activation_fn=tf.nn.relu)
            excitation = layers.fully_connected(inputs=excitation,
                                                num_outputs=squeeze.shape[-1].value, activation_fn=tf.nn.sigmoid)

            se_out = out * excitation
            avg_pool = layers.avg_pool2d(inputs=se_out, kernel_size=[2, 2], stride=2, padding='SAME')
            print(avg_pool)
        else:
            avg_pool = out

    return avg_pool


def dense_block(input_x, layer_name, rep, growth_k, is_training, use_dropout=False,
                use_se=False, reduction_ratio=16):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, growth_k, is_training, use_dropout)
        layers_concat.append(x)

        for i in range(rep - 1):
            x = tf.concat(layers_concat, axis=3)
            x = bottleneck_layer(x, growth_k, is_training, use_dropout)
            layers_concat.append(x)
        x = tf.concat(layers_concat, axis=3)

        if use_se:
            squeeze = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # global average pooling
            excitation = layers.fully_connected(inputs=squeeze,
                                                num_outputs=squeeze.shape[-1].value // reduction_ratio,
                                                activation_fn=tf.nn.relu)
            excitation = layers.fully_connected(inputs=excitation,
                                                num_outputs=squeeze.shape[-1].value, activation_fn=tf.nn.sigmoid)
            x = x * excitation
    return x


def load_logistic_coef(coef_path):
    with open(coef_path, 'r') as f:
        coef = json.load(f)
        weight_init = np.expand_dims(np.array(coef[0], dtype=np.float32), axis=-1)
        bias_init = np.array(coef[1], dtype=np.float32)
        return weight_init, bias_init


class InferenceModel63:
    def __init__(self, trainable=False, pre_model='InferenceModel58', pre_nn_model='InferenceModel33',
                 image_size=280, image_height=210, init_exp='exp151', pre_serial=0,
                 learning_rate=0.01, decay_steps=5000, decay_rate=0.94, decay=0.9, epsilon=0.1,
                 alpha=0.3, gamma=2, **kwargs):
        self.model_scope, self.classifier_scope = design_scope(class_name=pre_model)
        nn_scope = set_scope(pre_nn_model, 'NN')
        aggregate_scope = set_scope(type(self).__name__, 'Aggregate')

        self.img_h, self.img_w, self.img_c = image_height, image_size, 1
        self.class_num = 1

        # Hyper-parameter of cnn based on dense block
        growth_k, theta = 32, 0.5
        block_rep_list, k_p_list = [1,1,1,1], [1,1,1,1]
        block_num = len(block_rep_list)
        use_se=False

        self.images = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.img_c])
        self.labels = tf.placeholder(tf.int64, shape=[None])
        self.is_training = tf.placeholder(tf.bool, shape=None)

        with tf.variable_scope('Clinical'):
            with tf.name_scope('age'):
                self.ages = tf.placeholder('float32', [None, ])
                age = tf.reshape(self.ages, [-1, 1])

            with tf.name_scope('vas'):
                self.vas = tf.placeholder('float32', [None, ])
                vas = tf.reshape(self.vas, [-1, 1])

            with tf.name_scope('trauma'):
                self.tm = tf.placeholder('float32', [None, ])
                tm = tf.reshape(self.tm, [-1, 1])

            with tf.name_scope('dominant'):
                self.dm = tf.placeholder('float32', [None, ])
                dm = tf.reshape(self.dm, [-1, 1])

            with tf.name_scope('clv_nn'):
                self.clv = tf.concat([0.01*age, 0.1*vas, tm, dm], axis=1)
                print('clv: ', self.clv)

        features_2d = self.images
        first_conv = layers.conv2d(inputs=features_2d, num_outputs=2 * growth_k, kernel_size=[7, 7],
                                   stride=2, activation_fn=None)
        print('1st conv: ', first_conv)

        first_pool = layers.max_pool2d(inputs=first_conv, kernel_size=[3, 3], stride=2, padding='SAME')
        print('1st pool: ', first_pool)

        dsb = first_pool
        for i in range(0, block_num-1):
            dsb = dense_block(input_x=dsb, layer_name=self.model_scope + '_DB'+str(i+1), rep=block_rep_list[i],
                              growth_k=k_p_list[i]*growth_k, use_se=use_se, is_training=self.is_training)
            dsb = transition_layer(input_x=dsb, layer_name='Transition'+str(i+1),
                                   theta=theta, is_training=self.is_training)

        self.last_dsb = dense_block(input_x=dsb, layer_name=self.model_scope + '_DB'+str(block_num),
                               rep=block_rep_list[-1], growth_k=k_p_list[-1]*growth_k,
                               use_se=use_se, is_training=self.is_training)

        self.bn_relu = tf.nn.relu(layers.batch_norm(self.last_dsb,
                                                    center=True, scale=True, is_training=self.is_training))
        print('bn_relu: ', self.bn_relu)
        self.last_pool = tf.reduce_mean(self.bn_relu, axis=[1, 2], keepdims=True)  # global average pooling
        print('last_pool: ', self.last_pool)

        flatten = layers.flatten(self.last_pool)
        self.img_fc = layers.fully_connected(inputs=flatten, num_outputs=flatten.shape[-1].value,
                                             activation_fn=tf.nn.relu)
        self.img_logit = layers.fully_connected(inputs=self.img_fc, num_outputs=self.class_num, activation_fn=None)

        clv_coef_path = os.path.join('/data/SNUBH/SSC/', init_exp, 'npy', 'clv_lgs_weight.json')
        clv_weight_init, clv_bias_init = load_logistic_coef(clv_coef_path)

        mm_coef_path = os.path.join('/data/SNUBH/SSC/', 'exp153', self.model_scope,
                                    'result-%03d' % pre_serial, 'mm_pre_weight.json')
        mm_weight_init, mm_bias_init = load_logistic_coef(mm_coef_path)

        with tf.name_scope('NN'):
            with tf.variable_scope(nn_scope):
                self.nn_fc = self.clv
                self.clv_weight = tf.get_variable('l_w', initializer=clv_weight_init, dtype=tf.float32)
                self.clv_bias = tf.get_variable('l_b', initializer=clv_bias_init, dtype=tf.float32)
                print(self.clv_weight, self.clv_bias)
                self.nn_logit = tf.matmul(self.nn_fc, self.clv_weight) + self.clv_bias
                print('self.clv_lgt: ', self.nn_logit)

            with tf.variable_scope(aggregate_scope):
                self.mm_weight = tf.get_variable('m_w', initializer=mm_weight_init, dtype=tf.float32)
                self.mm_bias = tf.get_variable('m_b', initializer=mm_bias_init, dtype=tf.float32)
                print(self.mm_weight, self.mm_bias)
                self.img_clv = tf.concat([self.img_logit, self.nn_logit], axis=1)

                print('self.img_dl: ', self.img_clv)
                self.logits = tf.matmul(self.img_clv, self.mm_weight) + self.mm_bias

                print('logits: ', self.logits)
                self.prob = tf.nn.sigmoid(self.logits)
                print('prob: ', self.prob)

        labels = tf.expand_dims(self.labels, axis=-1)
        self.accuracy = calculate_accuracy(self.prob[:, 0], tf.cast(self.labels, dtype=tf.float32))

        focal_loss = focal_loss_sigmoid(labels=labels, logits=self.logits, alpha=alpha, gamma=gamma)
        self.loss = tf.reduce_mean(focal_loss)

        grad_con = tf.reduce_mean(tf.gradients(self.prob, self.bn_relu)[0], axis=[1, 2], keepdims=True)
        self.local1 = tf.reduce_mean(grad_con * self.bn_relu, axis=-1, keepdims=True)
        self.local = tf.image.resize_bilinear(images=self.local1, size=[self.img_h, self.img_w])
        print('local: ', self.local)

        if trainable:
            self.global_step, self.global_epoch, self.train = \
                training_option(self.loss, learning_rate=learning_rate, decay_steps=decay_steps,
                                decay_rate=decay_rate, decay=decay, epsilon=epsilon)


class InferenceModel64:
    def __init__(self, trainable=False, pre_model='InferenceModel58', pre_nn_model='InferenceModel33',
                 image_size=280, image_height=210, learning_rate=0.01, decay_steps=5000,
                 decay_rate=0.94, decay=0.9, epsilon=0.1, alpha=0.3, gamma=2, **kwargs):
        self.model_scope, self.classifier_scope = design_scope(class_name=pre_model)
        nn_scope = set_scope(pre_nn_model, 'NN')
        aggregate_scope = set_scope(type(self).__name__, 'Aggregate')

        self.img_h, self.img_w, self.img_c = image_height, image_size, 1
        self.class_num = 1

        # Hyper-parameter of cnn based on dense block
        growth_k, theta = 32, 0.5
        block_rep_list, k_p_list = [1,1,1,1], [1,1,1,1]
        block_num = len(block_rep_list)
        use_se=False

        self.images = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.img_c])
        self.labels = tf.placeholder(tf.int64, shape=[None])
        self.is_training = tf.placeholder(tf.bool, shape=None)

        with tf.variable_scope('Clinical'):
            with tf.name_scope('age'):
                self.ages = tf.placeholder('float32', [None, ])
                age = tf.reshape(self.ages, [-1, 1])

            with tf.name_scope('vas'):
                self.vas = tf.placeholder('float32', [None, ])
                vas = tf.reshape(self.vas, [-1, 1])

            with tf.name_scope('trauma'):
                self.tm = tf.placeholder('float32', [None, ])
                tm = tf.reshape(self.tm, [-1, 1])

            with tf.name_scope('dominant'):
                self.dm = tf.placeholder('float32', [None, ])
                dm = tf.reshape(self.dm, [-1, 1])

            with tf.name_scope('clv_nn'):
                self.clv = tf.concat([0.01*age, 0.1*vas, tm, dm], axis=1)
                print('clv: ', self.clv)

        features_2d = self.images
        first_conv = layers.conv2d(inputs=features_2d, num_outputs=2 * growth_k, kernel_size=[7, 7],
                                   stride=2, activation_fn=None)
        print('1st conv: ', first_conv)

        first_pool = layers.max_pool2d(inputs=first_conv, kernel_size=[3, 3], stride=2, padding='SAME')
        print('1st pool: ', first_pool)

        dsb = first_pool
        for i in range(0, block_num-1):
            dsb = dense_block(input_x=dsb, layer_name=self.model_scope + '_DB'+str(i+1), rep=block_rep_list[i],
                              growth_k=k_p_list[i]*growth_k, use_se=use_se, is_training=self.is_training)
            dsb = transition_layer(input_x=dsb, layer_name='Transition'+str(i+1),
                                   theta=theta, is_training=self.is_training)

        self.last_dsb = dense_block(input_x=dsb, layer_name=self.model_scope + '_DB'+str(block_num),
                               rep=block_rep_list[-1], growth_k=k_p_list[-1]*growth_k,
                               use_se=use_se, is_training=self.is_training)

        self.bn_relu = tf.nn.relu(layers.batch_norm(self.last_dsb,
                                                    center=True, scale=True, is_training=self.is_training))
        print('bn_relu: ', self.bn_relu)

        self.last_pool = tf.reduce_mean(self.bn_relu, axis=[1, 2], keepdims=True)  # global average pooling
        print('last_pool: ', self.last_pool)

        flatten = layers.flatten(self.last_pool)
        self.img_fc = layers.fully_connected(inputs=flatten, num_outputs=flatten.shape[-1].value,
                                             activation_fn=tf.nn.relu)
        self.img_logit = layers.fully_connected(inputs=self.img_fc, num_outputs=self.class_num, activation_fn=None)

        with tf.name_scope('NN'):
            with tf.variable_scope(nn_scope):
                self.nn_fc = self.clv
                self.clv_weight = tf.get_variable('l_w', [self.nn_fc.shape[-1].value, 1],
                                                  initializer=layers.xavier_initializer(), dtype=tf.float32)
                self.clv_bias = tf.get_variable('l_b', [1,], initializer=layers.xavier_initializer(), dtype=tf.float32)
                self.nn_logit = tf.matmul(self.nn_fc, self.clv_weight) + self.clv_bias
                print('self.clv_lgt: ', self.nn_logit)

            with tf.variable_scope(aggregate_scope):
                self.img_clv = tf.concat([self.img_logit, self.nn_logit], axis=1)
                self.mm_weight = tf.get_variable('m_w', [self.img_clv.shape[-1].value, 1],
                                                 initializer=layers.xavier_initializer(), dtype=tf.float32)
                self.mm_bias = tf.get_variable('m_b', [1,], initializer=layers.xavier_initializer(), dtype=tf.float32)

                print('self.img_dl: ', self.img_clv)
                self.logits = tf.matmul(self.img_clv, self.mm_weight) + self.mm_bias

                print('logits: ', self.logits)
                self.prob = tf.nn.sigmoid(self.logits)
                print('prob: ', self.prob)

        labels = tf.expand_dims(self.labels, axis=-1)
        self.accuracy = calculate_accuracy(self.prob[:, 0], tf.cast(self.labels, dtype=tf.float32))

        focal_loss = focal_loss_sigmoid(labels=labels, logits=self.logits, alpha=alpha, gamma=gamma)
        self.loss = tf.reduce_mean(focal_loss)

        grad_con = tf.reduce_mean(tf.gradients(self.prob, self.bn_relu)[0], axis=[1, 2], keepdims=True)
        self.local1 = tf.reduce_mean(grad_con * self.bn_relu, axis=-1, keepdims=True)
        self.local = tf.image.resize_bilinear(images=self.local1, size=[self.img_h, self.img_w])
        print('local: ', self.local)

        if trainable:
            self.global_step, self.global_epoch, self.train = \
                training_option(self.loss, learning_rate=learning_rate, decay_steps=decay_steps,
                                decay_rate=decay_rate, decay=decay, epsilon=epsilon)


def training_option(loss, learning_rate=0.001, decay_steps=1000, decay_rate=0.9, decay=0.9, epsilon=0.1):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

    with tf.variable_scope('reg_loss'):
        reg_loss = 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
    lr_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=tf.train.get_global_step(),
                                         decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate, decay=decay, epsilon=epsilon)
        train = optimizer.minimize(loss+reg_loss, global_step=tf.train.get_global_step())
    return global_step, global_epoch, train


if __name__ == '__main__':
    InferenceModel63()
    train_vars = tf.trainable_variables()
