import numpy as np
from models.model import GRU
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random

# GRU net
dim_in = 6
dim_out = 1
n_layer = 1
TIME_STEP = 15  # rnn time step
LR = 0.02  # learning rate
gru = GRU(dim_in, dim_out, n_layer)
print(gru)
optimizer = torch.optim.Adam(gru.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()
h_state = None  # for initial hidden state

# subframe
scaling = 1
MP = 200
MT = 40
number = 2
drx = 160
on_time = 8
inavtive_time = 100
p_c = 100 * scaling
p_cs = 300 * scaling
p_sleep = 1
p_tr = 450
sp_time = 0.5
speed = [0.5/26, 0.5 / 7, 0.5/5]
read_speed = 2.33 / 1000
buffer = 0
bool_scheme = 1
max_buffer = 70

bool_on = 0
bool_inactive = 0
bool_sleep = 0

bool_data = 1
bool_last = 0
bool_buffer = 0

power_av = 0
time_av = 0  # 平均时延

data_sup = 0
ue_buffer = 0

# show data
state_tr_matrix = np.array([[0.3, 0.1, 0.1], [0.6, 0.8, 0.5], [0.1, 0.1, 0.3]])
state = 1  # [0:5% SINR, 1:50% SINR, 2:95% SINR]
state_T = 1000

for i in range(MT):
    bs_buffer_record = []
    ue_buffer_record = []
    N_record = []

    on_timer = on_time
    avtive_timer = inavtive_time
    sleep_timer = drx
    t = 1
    power_sum = 0
    buffer_prediction = 0
    point = round(random.uniform(0, 1) * drx)

    state_list = []
    buffer_list = []
    drx_list = []  # [0:sleep , 1: avtive time]
    ue_buffer_list = []

    for j in range(number):
        N = round(6000 + random.uniform(-500, 500))
        N_record.append(N)
        x = np.random.poisson(lam=MP, size=N)
        gap = x[0]
        gap_times = 0

        delay = []
        buffer = buffer + sp_time
        while True:
            # 网络训练数据更新
            if t >= 1:
                state_list.insert(0, state)
                buffer_list.insert(0, buffer)
                if bool_inactive or bool_on:
                    drx_list.insert(0, 1)
                else:
                    drx_list.insert(0, 0)
                ue_buffer_list.insert(0, ue_buffer)
                if len(buffer_list) > TIME_STEP:
                    max_buffer_list = np.ones(TIME_STEP) * max_buffer
                    read_speed_list = np.ones(TIME_STEP) * read_speed

                    data = np.vstack((max_buffer_list, read_speed_list))
                    data = np.vstack((data, np.array(state_list[1:])))
                    data = np.vstack((data, np.array(buffer_list[1:])))
                    data = np.vstack((data, np.array(drx_list[1:])))
                    data = np.vstack((data, np.array(ue_buffer_list[1:])))

                    state_list.pop(-1)
                    buffer_list.pop(-1)
                    drx_list.pop(-1)
                    ue_buffer_list.pop(-1)

                    x_train = torch.from_numpy(data.T.reshape(1, data.T.shape[0], data.T.shape[1]))
                    x_train = torch.tensor(x_train, dtype=torch.float32)

                    y = np.array(buffer_list[-1]).reshape((1, 1))

                    prediction, h_state = gru(x_train, h_state)  # rnn output
                    # !! next step is important !!
                    h_state = h_state.data  # repack the hidden state, break the connection from last iteration
                    loss = loss_func(prediction, torch.tensor(y, dtype=torch.float32))  # calculate loss
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

            # 信道状态转移
            if t % state_T == 0:
                number = random.uniform(0, 1)
                cdf_prob = 0
                init_state = 0
                for prob in state_tr_matrix[:, state]:
                    cdf_prob = prob + cdf_prob
                    if number < cdf_prob:
                        state = init_state
                        break
                    else:
                        init_state = init_state + 1

            if t == 1:  # 开始时刻的状态判断
                if point > on_time:
                    bool_sleep = 1
                    sleep_timer = drx - point
                    delay.append(drx - point + sp_time)
                else:
                    bool_on = 1
                    on_timer = on_time - point
                    delay.append(sp_time)
                if bool_on:
                    bool_on = 0
                    on_timer = - 1

                    bool_data = 0

                    data_sup = buffer
                    avtive_timer = inavtive_time
                    bool_inactive = 1
            else:
                buffer_temp = buffer

                if gap == 0:  # 判断数据
                    # 判断到来的数据是否需要开启新的active timer
                    bool_data = 1
                    if bool_inactive and buffer > 0:
                        bool_data = 0
                    gap_times = gap_times + 1
                    if gap_times <= N - 1:
                        gap = x[gap_times]
                        buffer = buffer + sp_time
                # DRX
                if avtive_timer == 0:
                    bool_inactive = 0
                    avtive_timer = avtive_timer - 1

                    if bool_on == 0:
                        power_sum = power_sum + p_tr
                        bool_sleep = 1

                if sleep_timer == 0:
                    sleep_timer = drx

                    bool_on = 1
                    on_timer = on_time
                    bool_sleep = 0
                if bool_on or bool_inactive:  # Active time
                    if bool_inactive:
                        if bool_data:
                            bool_data = 0
                            data_sup = buffer
                            avtive_timer = inavtive_time
                        if data_sup > 0:
                            avtive_timer = inavtive_time
                    else:
                        if bool_on:
                            bool_sleep = 0
                            if buffer > 0:
                                data_sup = buffer
                                bool_data = 0
                                bool_inactive = 1
                                avtive_timer = inavtive_time

                if on_timer == 0:
                    bool_on = 0
                    on_timer = on_timer - 1
                    if bool_inactive == 0:
                        power_sum = power_sum + p_tr
                        bool_sleep = 1

            # buffer 策略
            if bool_scheme:
                if ue_buffer < max_buffer * 0.3 and bool_buffer == 1:
                    bool_buffer = 0
                if ue_buffer > max_buffer * 0.7 and bool_buffer == 0:
                    bool_buffer = 1

                if bool_buffer:
                    bool_on = 0
                    bool_inactive = 0
                    bool_sleep = 1

            if bool_on and bool_inactive != 1:
                power_sum = power_sum + p_c
            if bool_inactive:
                if data_sup > 0:
                    data_sup = data_sup - 1
                    if data_sup < 0:
                        data_sup = 0
                if buffer > 0 and ue_buffer + speed[state] * 2 < max_buffer:
                    buffer = buffer - speed[state] * 2
                    ue_buffer = ue_buffer + speed[state] * 2
                    if buffer > 0:
                        power_sum = power_sum + p_cs
                    else:
                        buffer = 0
                        power_sum = power_sum + (p_c + p_cs) * 0.5
                else:
                    power_sum = power_sum + p_c
            if bool_sleep:
                power_sum = power_sum + p_sleep
            # 时间的推进和定时器的减少
            bs_buffer_record.append(buffer)
            ue_buffer_record.append(ue_buffer)

            t = t + 1
            # AI buffer_prediction
            buffer_pre_temp = buffer_prediction

            if ue_buffer > 0:
                ue_buffer = ue_buffer - read_speed
                if ue_buffer < 0:
                    ue_buffer = 0

            if bool_on:
                on_timer = on_timer - 1
            if bool_inactive:
                avtive_timer = avtive_timer - 1
            sleep_timer = sleep_timer - 1

            gap = gap - 1
            if ue_buffer == 0 and gap_times >= N - 1:
                break

        power_av = power_sum / t + power_av
        print('蒙特卡洛次数：', i, ', 累积能量: ', power_av)

power_result = power_av / MT + 100 / 320
