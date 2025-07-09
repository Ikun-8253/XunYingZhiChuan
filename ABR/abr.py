# add queuing delay into halo
import os
import numpy as np
import abrenv
import load_trace

# new: delay, jitter, packet_loss, video_bitrate
S_INFO = 4
S_LEN = 5  # take how many frames in the past
A_DIM = 6
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 200
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


class ABREnv():

    def __init__(self, agent_id = 0, trace_dir = None):
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()

        self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                        all_cooked_bw=all_cooked_bw)

        self.last_bit_rate = DEFAULT_QUALITY
        self.buffer_size = 0.
        self.state = np.zeros((S_INFO, S_LEN))
        # self.reset()
        
    def seed(self, num):
        np.random.seed(num)

    def reset(self):
        # self.net_env.reset_ptr()
        self.time_stamp = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_size = 0.
        bit_rate = self.last_bit_rate

        delay, jitter, packet_loss, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate)

        state = np.roll(self.state, -1, axis=1) # 状态数组左移一列

        delay = float(delay) - abrenv.LINK_RTT
        video_bitrate = VIDEO_BIT_RATE[bit_rate]

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

        self.state = state
        return state
        #return state.reshape((1, S_INFO*S_LEN))  4行未知列数组

    def render(self):
        return

    def step(self, action):
        bit_rate = int(action)
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, jitter, packet_loss,  sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = self.net_env.get_video_chunk(bit_rate)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # 奖励函数 R=质量因子*视频质量（码率）- 平滑因子*视频流畅度 - 延迟因子*视频延迟
        reward = BITRATE_FACTOR * VIDEO_BIT_RATE[bit_rate] / M_IN_K + \
                 SMOOTH_FACTOR * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K - \
                 DELAY_FACTOR * delay / 1000

        reward /= 100.
        
        self.last_bit_rate = bit_rate
        state = np.roll(self.state, -1, axis=1)

        delay = float(delay) - abrenv.LINK_RTT
        video_bitrate = VIDEO_BIT_RATE[bit_rate]

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

        self.state = state
        # observation, reward, done, info = env.step(action)
        return state, reward, end_of_video, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf}
