import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 控制TensorFlow的日志只会输出错误（Error）信息
import tensorflow.compat.v1 as tf
import load_trace
import ppo2 as network
import fixed_env as env
tf.get_logger().setLevel('ERROR') # 屏蔽警告

S_INFO = 4  # delay, jitter, packet_loss, video_bitrate
S_LEN = 5   # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = np.array([5000, 10000, 15000, 20000, 25000, 30000])  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 157.0
M_IN_K = 1000.0
# REBUF_PENALTY = 160.0  # 1 sec rebuffering -> 3 Mbps

# 奖赏函数权重因子
BITRATE_FACTOR = 8
SMOOTH_FACTOR = 3
DELAY_FACTOR = 12

DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
# LOG_FILE = './test_results/log_sim_rl'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = sys.argv[1]
LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = './test/'
# LOG_FILE = sys.argv[3] + 'log_sim_rl'
# TEST_TRACES = sys.argv[2]
    
def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    with tf.Session() as sess:

        # 创建一个神经网络模型
        actor = network.Network(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer()) # 初始化tensorflow全局变量
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters 恢复神经网络参数
        if NN_MODEL is not None:  # NN_MODEL is the path to file 说明存在模型路径
            saver.restore(sess, NN_MODEL) # 加载预训练模型到当前tensorflow会话中
            print("Testing model restored.") # 模型加载成功

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1 # 后续检查这里bit_rate是否正确

        s_batch = [np.zeros((S_INFO, S_LEN))] # 存放state数据,state矩阵大小为S_INFO（参数量）*S_LEN（状态序列长度）
        a_batch = [action_vec] # 存放action数据,一个包含6个码率等级选择值的列表，范围1-6
        r_batch = [] # 存放reward数据,每个时间步的奖励
        entropy_record = [] # 存放策略网络的熵值,强化学习中,熵被用来衡量策略的不确定性
        entropy_ = 0.5
        video_count = 0
        
        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, jitter, packet_loss, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms
            video_bitrate = VIDEO_BIT_RATE[bit_rate]

            # 奖励函数 R=质量因子*视频质量（码率）- 平滑因子*视频流畅度 - 延迟因子*视频延迟
            reward = BITRATE_FACTOR * VIDEO_BIT_RATE[bit_rate] / M_IN_K + \
                     SMOOTH_FACTOR * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K - \
                     DELAY_FACTOR * delay / 1000

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # log_format: time_stamp(s), delay(ms), jitter(ms), packet_loss, video_bitrate(KB/s), entropy, reward
            log_file.write(str("{:.2f}".format(time_stamp / M_IN_K)) + '\t' +
                           str("{:.4f}".format(delay)) + '\t' +
                           str("{:.4f}".format(jitter)) + '\t' +
                           str("{:.6f}".format(packet_loss)) + '\t' +
                           str(video_bitrate) + '\t' +
                           str("{:.6f}".format(entropy_)) + '\t' +
                           str("{:.6f}".format(reward)) + '\n')
            log_file.flush()

            # retrieve previous state 获取先前状态
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1) # 数组元素循环左移一位,实现状态序列更新,例如 123456->234567

            delay = float(delay) - env.LINK_RTT

            # this should be S_INFO number of terms
            # 归一化↓ 每一行的各元素/每一行元素的最大值
            if np.max(state[0, :]) * np.max(state[1, :]) * np.max(state[2, :]) * np.max(state[1, :]) < 1e-6:
                state[0, -1] = 1.0
                state[1, -1] = 1.0
                state[2, -1] = 1.0
                state[3, -1] = 1.0
            else:
                state[0, -1] = float(delay) / float(np.max(state[0, :]))
                state[1, -1] = float(jitter) / float(np.max(state[1, :]))
                state[2, -1] = float(packet_loss) / float(np.max(state[2, :]))
                state[3, -1] = float(video_bitrate) / float(np.max(state[3, :]))

            action_prob = actor.predict(state) # 根据states预测action,action_prob是动作概率向量，代表每个动作的概率
            
            noise = np.random.gumbel(size=len(action_prob)) # 生成一个服从 Gumbel 分布的随机噪声向量
            bit_rate = np.argmax(np.log(action_prob) + noise) # 将动作概率向量取对数再加上生成的随机噪声向量
            # 引入了一定程度的随机性,使预测动作的过程更多样化和鲁棒
            # 选择具有最大值的索引作为最终的动作（最大加权概率的动作）

            s_batch.append(state)
            entropy_ = -np.dot(action_prob, np.log(action_prob)) # 计算action_prob的熵值,用来度量智能体在当前状态下对动作的不确定性程度
            entropy_record.append(entropy_) # 将熵值entropy_加入entropy_record列表

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                # 清空列表
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM) # 创建新的动作向量
                action_vec[bit_rate] = 1 # 下一个码率动作

                s_batch.append(np.zeros((S_INFO, S_LEN))) # 更新状态
                a_batch.append(action_vec) # 更新动作
                # print(np.mean(entropy_record))
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
