import tensorflow as tf
import numpy as np
import sys, re
import os, json
from tensorflow.contrib import layers
sys.path.append('/home/chzze/bitbucket/SSC_rls')


def design_scope(class_name):
    model_scope = re.sub('Inference', '', class_name)
    classifier_scope = re.sub('Model', 'Classifier', model_scope)
    return model_scope, classifier_scope


def calculate_accuracy(prob, label):
    predicted = tf.cast(prob > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, label), dtype=tf.float32))
    return accuracy


def focal_loss_sigmoid(labels, logits, alpha=0.25, gamma=2):
    """
    Compute focal loss for binary classification
    Args:
    :param labels: A int32 tensor of shape [batch_size].
    :param logits: A float 32 tensor of shape [batch_size].
    :param alpha: A scalar for focal loss alpha hyper-parameter. If positive sample number
                  > negative sample number, alpha < 0.5 and vice versa.
    :param gamma: A scalar for focal loss gamma hyper-parameter.
    :return: A tensor of the same shape as 'labels'
    """
    y_pred = tf.nn.sigmoid(logits)
    labels = tf.to_float(labels)
    fcl_loss = -labels*(1-alpha)*((1-y_pred)**gamma)*tf.log(y_pred)-(1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    return fcl_loss


def get_global_vars(scope_list):
    _vars = []
    for scope in scope_list:
        _vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return _vars


def get_prep_conv_vars(scope1, scope2):
    scope_list = [scope1, scope2]
    _vars = get_global_vars(scope_list)
    return _vars


def get_prep_conv_train_vars(scope1, scope2):
    scope_list = [scope1, scope2]
    _vars = []
    for scope in scope_list:
        _vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return _vars


def get_train_info(data_path, key):
    with open(os.path.join(data_path, '.info'), 'r') as f:
        info = json.load(f)
        return info[key]


def conv_layer(inputs, in_channels, out_channels, is_training):
    conv = layers.conv2d(inputs=inputs, num_outputs=in_channels, kernel_size=1, stride=1, activation_fn=None)
    conv = tf.nn.relu(layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training))

    conv = layers.conv2d(inputs=conv, num_outputs=in_channels, kernel_size=3, stride=1, activation_fn=None)
    conv = tf.nn.relu(layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training))

    conv = layers.conv2d(inputs=conv, num_outputs=out_channels, kernel_size=1, stride=1, activation_fn=None)
    conv = layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training)
    return conv


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
                                                num_outputs=squeeze.shape[-1].value,
                                                activation_fn=tf.nn.sigmoid)
            x = x * excitation
    return x


class InferenceModel58:
    def __init__(self, trainable=False, growth_k=32, image_size=280, image_height=210,
                 learning_rate=0.01, decay_steps=7500, decay_rate=0.94, decay=0.9, epsilon=0.1, theta=0.5,
                 alpha=0.3, gamma=2, block_rep='1,1,1,1', k_p='1,1,1,1', use_se=False, **kwargs):

        self.model_scope, _ = design_scope(class_name=type(self).__name__)
        self.img_h, self.img_w, self.img_c = image_height, image_size, 1
        self.class_num = 1
        self.images = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.img_c])
        self.labels = tf.placeholder(tf.int64, shape=[None])
        self.is_training = tf.placeholder(tf.bool, shape=None)
        self.growth_k = growth_k

        block_rep_list = list(map(int, re.split(',', block_rep)))
        block_num = len(block_rep_list)
        k_p_list = list(map(int, re.split(',', k_p)))

        features_2d = self.images
        first_conv = layers.conv2d(inputs=features_2d, num_outputs=2 * self.growth_k, kernel_size=[7, 7],
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
        self.last_pool = tf.reduce_mean(self.bn_relu, axis=[1, 2], keepdims=True)  # global average pooling
        print('last_pool: ', self.last_pool)

        flatten = layers.flatten(self.last_pool)
        self.fc = layers.fully_connected(inputs=flatten, num_outputs=flatten.shape[-1].value,
                                         activation_fn=tf.nn.relu)
        self.logits = layers.fully_connected(inputs=self.fc, num_outputs=self.class_num,
                                             activation_fn=None)
        self.prob = tf.nn.sigmoid(self.logits)

        grad_con = tf.reduce_mean(tf.gradients(self.prob, self.bn_relu)[0], axis=[1, 2], keepdims=True)
        self.local1 = tf.reduce_mean(grad_con * self.bn_relu, axis=-1, keepdims=True)
        self.local = tf.image.resize_bilinear(images=self.local1, size=[self.img_h, self.img_w])
        print('local: ', self.local)

        labels = tf.expand_dims(self.labels, axis=-1)
        self.accuracy = calculate_accuracy(self.prob[:, 0], tf.cast(self.labels, dtype=tf.float32))
        focal_loss = focal_loss_sigmoid(labels=labels, logits=self.logits, alpha=alpha, gamma=gamma)
        self.loss = tf.reduce_mean(focal_loss)

        if trainable:
            self.global_step, self.global_epoch, self.train = \
                training_option(self.loss, learning_rate=learning_rate, decay_steps=decay_steps,
                                decay_rate=decay_rate, decay=decay, epsilon=epsilon)


def each_param(train_var):
    var_shape = train_var.shape.as_list()
    var_param = np.prod(np.array(var_shape))
    return var_param


def training_option(loss, learning_rate=0.01, decay_steps=5000, decay_rate=0.94, decay=0.9, epsilon=0.1):
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
    InferenceModel58(growth_k=32, image_size=280, image_height=210, block_rep='1,1,1,1', theta=0.5, use_se=False)

    train_vars = tf.trainable_variables()
    total_params = list(map(each_param, train_vars))

    print(train_vars)
    print(len(train_vars))
    print(sum(total_params))

