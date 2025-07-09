import math
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 控制TensorFlow的日志只会输出错误（Error）信息
import tensorflow.compat.v1 as tf
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tflearn
tf.get_logger().setLevel('ERROR') # 屏蔽警告

# FEATURE_NUM = 128 # 神经网络中每一层的特征数目   神经元
ACTION_EPS = 1e-4 # 动作的最小概率,用于在策略中进行裁剪
GAMMA = 0.99 # 折扣因子,一个介于0和1之间的实数,用于权衡当前奖励和未来奖励的重要性
# 当γ接近1时，代理更多地关注未来奖励，即更长期的回报。而当γ接近0时，代理更多地关注当前奖励，即更短期的回报

# PPO2
EPS = 0.2 # PPO算法中的超参数，用于限制策略更新的幅度,截断范围

class Network():
    def CreateNetwork(self, inputs):

        inputs_flat = tflearn.flatten(inputs)  # inputs是一个多维矩阵,送入全连接层前先进行flat展平处理

        with tf.variable_scope('actor'):
            fc_1 = tflearn.fully_connected(inputs_flat, 128, activation='linear')
            fc_1 = tf.nn.leaky_relu(fc_1, alpha=0.01)
            fc_2 = tflearn.fully_connected(fc_1, 64, activation='linear')
            fc_2 = tf.nn.leaky_relu(fc_2, alpha=0.01)
            pi = tflearn.fully_connected(fc_2, self.a_dim, activation='softmax')  # softmax将输出的原始分数转换成概率分布
            # 使得每个动作的输出值在 (0, 1) 区间内，并且所有动作的输出值之和为 1，表示概率

        with tf.variable_scope('critic'):
            fc_1 = tflearn.fully_connected(inputs_flat, 128, activation='linear')
            fc_1 = tf.nn.leaky_relu(fc_1, alpha=0.01)
            fc_2 = tflearn.fully_connected(fc_1, 64, activation='linear')
            fc_2 = tf.nn.leaky_relu(fc_2, alpha=0.01)
            value = tflearn.fully_connected(fc_2, 1, activation='linear')

        return pi, value
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    # 计算优势比率=新策略采取动作概率之和/旧策略采取动作概率之和
    def r(self, pi_new, pi_old, acts):
        return tf.reduce_sum(tf.multiply(pi_new, acts), reduction_indices=1, keepdims=True) / \
                tf.reduce_sum(tf.multiply(pi_old, acts), reduction_indices=1, keepdims=True)

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self._entropy = 5.
        self.quality = 0
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self.R = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.old_pi = tf.placeholder(tf.float32, [None, self.a_dim])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.entropy_weight = tf.placeholder(tf.float32)
        self.pi, self.val = self.CreateNetwork(inputs=self.inputs)
        self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)
        self.log_prob = tf.log(tf.reduce_sum(tf.multiply(self.real_out, self.acts), reduction_indices=1, keepdims=True))
        self.entropy = tf.multiply(self.real_out, tf.log(self.real_out))
        self.adv = tf.stop_gradient(self.R - self.val)
        self.ppo2loss = tf.minimum(self.r(self.real_out, self.old_pi, self.acts) * self.adv, 
                            tf.clip_by_value(self.r(self.real_out, self.old_pi, self.acts), 1 - EPS, 1 + EPS) * self.adv
                        ) # 计算ppo2的损失函数
        self.dual_loss = tf.cast(tf.less(self.adv, 0.), dtype=tf.float32)  * \
            tf.maximum(self.ppo2loss, 3. * self.adv) + \
            tf.cast(tf.greater_equal(self.adv, 0.), dtype=tf.float32) * \
            self.ppo2loss # 计算ppo的双重损失,根据优势的正负来选择使用 PPO 损失还是使用一个较大的损失
        
        self.a2closs = self.log_prob * self.adv # 计算Actor-Critic算法中的损失,它是对数概率和优势的乘积,用于优化策略网络
        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.network_params += \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        
        self.loss = - tf.reduce_sum(self.dual_loss) \
            + self.entropy_weight * tf.reduce_sum(self.entropy) # 总损失，包括策略损失和熵正则化项的加权和。
                                                     # 策略损失是双重损失函数，熵正则化项用于促进策略的多样性
        
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss) # 使用Adam优化器最小化总损失[优化操作]
        self.val_loss = tflearn.mean_square(self.val, self.R) # 定义了值函数的均方误差损失
        self.val_opt = tf.train.AdamOptimizer(self.lr_rate * 10.).minimize(self.val_loss) # 使用学习率是原始学习率的10倍的
                                                                                   # Adam优化器最小化值函数的均方误差损失[优化操作]

        # 新加的loss曲线绘制代码
        self.ppo2loss_summary = tf.summary.scalar('ppo2loss', tf.reduce_mean(self.ppo2loss))
        self.dual_loss_summary = tf.summary.scalar('dual_loss', tf.reduce_mean(self.dual_loss))
        self.a2closs_summary = tf.summary.scalar('a2closs', tf.reduce_mean(self.a2closs))
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.val_loss_summary = tf.summary.scalar('val_loss', self.val_loss)


    # 神经网络进行前向传播，得到输出动作的概率分布,通过 feed_dict 参数将输入状态传递给神经网络
    def predict(self, inputs):
        inputs = np.reshape(inputs, (-1, self.s_dim[0], self.s_dim[1]))
        action = self.sess.run(self.real_out, feed_dict={
            self.inputs: inputs
        })
        return action[0] # 返回的是计算得到的动作概率分布中的第一个动作,即预测的下一步要采取的动作
    
    def set_entropy_decay(self, decay=0.6):
        self._entropy *= decay # 对当前熵的值 _entropy 进行衰减操作，具体操作是将当前熵乘以衰减率 decay，从而减小熵的值
                          # 用于控制策略的探索程度，在训练过程中逐渐降低探索的程度，以便模型更加专注于利用当前的策略

    def get_entropy(self, step):
        return np.clip(self._entropy, 1e-1, 5.) # 获取当前熵的值，并可能进行裁剪以确保熵的值在一个合理的范围内[1e-1, 5.]

    # 神经网络将根据输入的批量数据进行前向传播和反向传播，然后通过优化算法（Adam）更新网络参数，
    # 以最小化损失函数（包括策略损失和值函数损失）并提高网络的性能
    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):
        s_batch, a_batch, p_batch, v_batch = tflearn.data_utils.shuffle(s_batch, a_batch, p_batch, v_batch)
        self.sess.run([self.optimize, self.val_opt], feed_dict={
            self.inputs: s_batch,
            self.acts: a_batch,
            self.R: v_batch, 
            self.old_pi: p_batch,
            self.entropy_weight: self.get_entropy(epoch)
        })

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        ba_size = len(s_batch)
        R_batch = np.zeros([len(r_batch), 1])

        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:    
            v_batch = self.sess.run(self.val, feed_dict={
                self.inputs: s_batch
            })
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        return list(R_batch)
