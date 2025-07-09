import multiprocessing as mp
import numpy as np
import logging
import os
import sys
from abr import ABREnv
import ppo2 as network
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 控制TensorFlow的日志只会输出错误（Error）信息
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR') # 屏蔽警告
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

S_DIM = [4, 5]
A_DIM = 6
ACTOR_LR_RATE =1e-4
CRITIC_LR_RATE = 1e-3
NUM_AGENTS = 8
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 1000000
MODEL_SAVE_INTERVAL = 200
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './rl-5g'
TRAIN_TRACES = './train/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = './rl-5g/log'
PPO_TRAINING_EPO = 5
# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None    

def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    #os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system('python rl_test.py ' + nn_model)

    # append test performance to the log
    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        entropies.append(np.mean(entropy[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards) # 奖励最小值
    rewards_5per = np.percentile(rewards, 5) # 5%分位数
    rewards_mean = np.mean(rewards) # 平均值
    rewards_median = np.percentile(rewards, 50) # 中位数
    rewards_95per = np.percentile(rewards, 95) # 95%分位数
    rewards_max = np.max(rewards) # 奖励最大值

    log_file.write(str(epoch) + '\t' +
                   str("{:.6f}".format(rewards_min)) + '\t' +
                   str("{:.6f}".format(rewards_5per)) + '\t' +
                   str("{:.6f}".format(rewards_mean)) + '\t' +
                   str("{:.6f}".format(rewards_median)) + '\t' +
                   str("{:.6f}".format(rewards_95per)) + '\t' +
                   str("{:.6f}".format(rewards_max)) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies) # 返回奖励平均值和熵的平均值
        
def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    tf_config=tf.ConfigProto(intra_op_parallelism_threads=5,
                            inter_op_parallelism_threads=5)
    with tf.Session(config = tf_config) as sess, open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        summary_ops, summary_vars = build_summaries()

        actor = network.Network(sess,
                state_dim=S_DIM, action_dim=A_DIM,
                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")
        
        max_reward, max_epoch = -10000., 0
        tick_gap = 0
        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, g = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, g_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                g += g_
            s_batch = np.stack(s, axis=0) # 状态
            a_batch = np.vstack(a) # 动作
            p_batch = np.vstack(p) # 策略 （预测）
            v_batch = np.vstack(g) # 回报 （价值）

            for _ in range(PPO_TRAINING_EPO):
                actor.train(s_batch, a_batch, p_batch, v_batch, epoch)
            # actor.train(s_batch, a_batch, v_batch, epoch)

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                avg_reward, avg_entropy = testing(epoch,
                    SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
                    test_log_file)

                if avg_reward > max_reward:
                    max_reward = avg_reward
                    max_epoch = epoch
                    tick_gap = 0
                else:
                    tick_gap += 1
                
                if tick_gap >= 5:
                    # saver.restore(sess, SUMMARY_DIR + "/nn_model_ep_" + str(max_epoch) + ".ckpt")
                    actor.set_entropy_decay()
                    tick_gap = 0

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: actor.get_entropy(epoch),
                    summary_vars[1]: avg_reward,
                    summary_vars[2]: avg_entropy
                })
                writer.add_summary(summary_str, epoch)

                # 新加的loss曲线绘制代码
                ppo2loss_summary_str, dual_loss_summary_str, a2closs_summary_str, loss_summary_str, val_loss_summary_str = sess.run(
                    [actor.ppo2loss_summary, actor.dual_loss_summary, actor.a2closs_summary, actor.loss_summary,
                     actor.val_loss_summary],
                    feed_dict={
                        actor.inputs: s_batch,
                        actor.acts: a_batch,
                        actor.R: v_batch,
                        actor.old_pi: p_batch,
                        actor.entropy_weight: actor.get_entropy(epoch)
                    })

                writer.add_summary(ppo2loss_summary_str, epoch)
                writer.add_summary(dual_loss_summary_str, epoch)
                writer.add_summary(a2closs_summary_str, epoch)
                writer.add_summary(loss_summary_str, epoch)
                writer.add_summary(val_loss_summary_str, epoch)

                writer.flush() # 立即将缓冲区中的所有数据写入到磁盘

def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id)
    with tf.Session() as sess, open(SUMMARY_DIR + '/log_agent_' + str(agent_id), 'w') as log_file:
        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

        time_stamp = 0

        for epoch in range(TRAIN_EPOCH):
            obs = env.reset()
            s_batch, a_batch, p_batch, r_batch = [], [], [], []
            for step in range(TRAIN_SEQ_LEN):
                s_batch.append(obs)

                action_prob = actor.predict(
                    np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

                # action_cumsum = np.cumsum(action_prob)
                # bit_rate = (action_cumsum > np.random.randint(
                #    1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # gumbel noise
                noise = np.random.gumbel(size=len(action_prob))
                bit_rate = np.argmax(np.log(action_prob) + noise)

                obs, rew, done, info = env.step(bit_rate)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)
                r_batch.append(rew)
                p_batch.append(action_prob)
                if done:
                    break
            v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
            exp_queue.put([s_batch, a_batch, p_batch, v_batch])

            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)

def build_summaries():
    td_loss = tf.Variable(0.)  # TD（时间差分）损失
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)  # 每个episode 的总奖励
    tf.summary.scalar("Reward", eps_total_reward)
    entropy = tf.Variable(0.)
    tf.summary.scalar("Entropy", entropy)

    summary_vars = [td_loss, eps_total_reward, entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def main():

    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
