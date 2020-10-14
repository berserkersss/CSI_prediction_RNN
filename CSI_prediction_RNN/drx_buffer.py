import random
import numpy as np

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
speed = 0.5 / 7
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

for i in range(MT):
    bs_buffer_record = []
    ue_buffer_record = []
    N_record = []

    on_timer = on_time
    avtive_timer = inavtive_time
    sleep_timer = drx
    t = 1
    power_sum = 0
    point = round(random.uniform(0, 1) * drx)

    for j in range(number):
        N = round(6000 + random.uniform(-500, 500))
        N_record.append(N)
        x = np.random.poisson(lam=MP, size=N)
        gap = x[0]
        gap_times = 0

        delay = [] 
        buffer = buffer + sp_time 
        while True:
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
                if buffer > 0 and ue_buffer + speed * 2 < max_buffer:
                    buffer = buffer - speed * 2
                    ue_buffer = ue_buffer + speed * 2
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
        print('蒙特卡洛次数：', i, ', 累积能量: ',  power_av)

power_result = power_av / MT + 100 / 320
