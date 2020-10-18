import numpy as np
from models.model import GRU
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random
from queue import Queue

# subframe
scaling = 1
MP = 200
MT = 20
n_video = 2
drx = 160
on_time = 8
inavtive_time = 100
p_c = 100 * scaling
p_cs = 300 * scaling
p_sleep = 1
p_tr = 450
packe_size = 0.5
speed = [0.5 / 26, 0.5 / 7, 0.5 / 5]
read_speed = 2.33 / 1000
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
break_av = 0

ue_buffer = 0

bs_queue = Queue(maxsize=0)  # 用于存储到达的包，每个0.5MB
pack_queue = Queue(maxsize=0)  # 用于存储到达的包的时间

# show data
state_tr_matrix = np.array([[0.3, 0.1, 0.1], [0.6, 0.8, 0.5], [0.1, 0.1, 0.3]])
state = 1  # [0:5% SINR, 1:50% SINR, 2:95% SINR]
state_T = 1000

loss_record = []
pre_buffer_record = []
rel_buffer_record = []

active_record = []
sleep_record = []
power_record = []
data_sup_record = []

for i in range(MT):
    print(i)
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

    data_sup = 0
    delay = []
    break_time = 0
    x_boool = 0
    for j in range(n_video):
        N = round(6000 + random.uniform(-500, 500))
        N_record.append(N)
        x = np.random.poisson(lam=MP, size=N)
        gap = x[0]
        gap_times = 0

        bs_queue.put(packe_size)
        pack_queue.put(t)
        time_x = 3200
        while True:
            power_x = power_sum
            # 信道状态转移
            # if t % state_T == 0:
            #     number = random.uniform(0, 1)
            #     cdf_prob = 0
            #     init_state = 0
            #     for prob in state_tr_matrix[:, state]:
            #         cdf_prob = prob + cdf_prob
            #         if number < cdf_prob:
            #             state = init_state
            #             break
            #         else:
            #             init_state = init_state + 1
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

            if t == 1:  # 开始时刻的状态判断
                if point > on_time:
                    bool_sleep = 1
                    sleep_timer = drx - point

                else:
                    bool_on = 1
                    on_timer = on_time - point

                if bool_on:
                    bool_on = 0
                    on_timer = - 1

                    bool_data = 0

                    if not bs_queue.empty():
                        data_sup = bs_queue.get()

                    avtive_timer = inavtive_time
                    bool_inactive = 1
            else:
                if gap == 0:  # 判断数据
                    # 判断到来的数据是否需要开启新的active timer
                    bool_data = 1
                    if bool_inactive and data_sup > 0:
                        bool_data = 0
                    gap_times = gap_times + 1
                    if gap_times <= N - 1:
                        gap = x[gap_times]
                        bs_queue.put(packe_size)
                        pack_queue.put(t)
                # DRX
                if avtive_timer == 0:
                    bool_inactive = 0
                    avtive_timer = avtive_timer - 1

                    if bool_on == 0 and bool_buffer == 0:
                        power_sum = power_sum + p_tr
                        bool_sleep = 1

                if sleep_timer == 0:
                    sleep_timer = drx

                    if bool_buffer == 0:
                        bool_on = 1
                        on_timer = on_time
                        bool_sleep = 0

                if (bool_on or bool_inactive) and (bool_buffer == 0):  # Active time
                    if bool_inactive:
                        if bool_data:
                            bool_data = 0
                            if not bs_queue.empty():
                                data_sup = bs_queue.get()
                            avtive_timer = inavtive_time
                        if data_sup > 0:
                            avtive_timer = inavtive_time
                        else:
                            if not bs_queue.empty():
                                data_sup = bs_queue.get()
                                avtive_timer = inavtive_time
                    else:
                        if bool_on:
                            bool_sleep = 0
                            if not bs_queue.empty():
                                if data_sup == 0:
                                    data_sup = bs_queue.get()
                            if data_sup > 0:
                                bool_data = 0
                                bool_inactive = 1
                                avtive_timer = inavtive_time

                if on_timer == 0:
                    bool_on = 0
                    on_timer = on_timer - 1
                    if bool_inactive == 0 and bool_buffer == 0:
                        power_sum = power_sum + p_tr
                        bool_sleep = 1

            if bool_on and bool_inactive != 1:
                power_sum = power_sum + p_c
            if bool_inactive:
                if data_sup > 0:
                    if ue_buffer + speed[state] * 2 < max_buffer:
                        ue_buffer = ue_buffer + speed[state] * 2
                    data_sup = data_sup - speed[state] * 2
                    if data_sup < min(speed) * 2:
                        data_sup = 0
                        delay.append(t - pack_queue.get())
                    power_sum = power_sum + p_cs
                else:
                    power_sum = power_sum + p_c
            if bool_sleep:
                power_sum = power_sum + p_sleep
            # 时间的推进和定时器的减少
            bs_buffer_record.append(bs_queue.qsize() + data_sup)
            ue_buffer_record.append(ue_buffer)

            t = t + 1
            # AI buffer_prediction
            buffer_pre_temp = buffer_prediction

            if ue_buffer > 0:
                ue_buffer = ue_buffer - read_speed
                if ue_buffer < 0:
                    ue_buffer = 0
                    break_time = break_time + 1

            if bool_on:
                on_timer = on_timer - 1
            if bool_inactive:
                avtive_timer = avtive_timer - 1
            sleep_timer = sleep_timer - 1

            gap = gap - 1
            if ue_buffer == 0 and gap_times >= N - 1 and pack_queue.empty():
                break

            # test
            # if bool_buffer == 1:
            #     x_boool = 1
            #
            # if x_boool == 1:
            #     time_x = time_x - 1
            #     if (bool_inactive or bool_on) and bool_buffer == 0:
            #         active_record.append(1)
            #     else:
            #         active_record.append(0)
            #     sleep_record.append(bool_buffer)
            #     power_record.append(power_sum)
            #     data_sup_record.append(data_sup)

            # if time_x == 0:
            #     print(j)
            #     plt.figure()
            #     plt.plot(active_record, c='r', label='Active')
            #     plt.plot(data_sup_record, c='green', label='Data_sup')
            #     plt.xlabel('T')
            #     plt.ylabel('value')
            #     plt.legend()
            #     plt.savefig("active.png")
            #     plt.show()
            #
            #     plt.figure()
            #     plt.plot(sleep_record, c='r', label='buffer')
            #     plt.xlabel('T')
            #     plt.ylabel('value')
            #     plt.legend()
            #     plt.savefig("buffer.png")
            #     plt.show()
            #
            #     plt.figure()
            #     plt.plot(power_record, c='r', label='power_sum')
            #     plt.xlabel('T')
            #     plt.ylabel('value')
            #     plt.legend()
            #     plt.savefig("power.png")
            #     plt.show()
            #     print("sf")

        power_av = power_sum / t + power_av
        #print(power_av)
        time_av = sum(delay) / N + time_av
        #print(time_av)
        break_av = break_time / t + break_av

    if i == 1:
        print(N_record)
        plt.figure()
        plt.plot(bs_buffer_record, c='r', label='BS_buffer')
        plt.plot(ue_buffer_record, c='b', label='UE_buffer')
        plt.xlabel('T/ms')
        plt.ylabel('Size/MB')
        plt.legend()
        plt.savefig("buffer.png")
        plt.show()

power_av = power_av / MT + 100 / 320
print("能耗：", power_av)
time_av = time_av / MT
print("时延", time_av)
break_av = break_av / MT
print("中断概率", break_av)
