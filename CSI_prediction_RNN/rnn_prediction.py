"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
numpy
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 1  # rnn time step
INPUT_SIZE = 1  # rnn input size
LR = 0.02  # learning rate

# show data
state_tr_matrix = np.array([[0.3, 0.1, 0.1], [0.6, 0.8, 0.5], [0.1, 0.1, 0.3]])
state = 0
csi = []
for step in range(2000):
    csi.append(state)
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

csi = np.array(csi)
# steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32) # float32 for converting torch FloatTensor
steps = csi
x_np = csi[:-1].astype(np.float32)
y_np = csi[1:].astype(np.int64)

plt.plot(np.arange(x_np.size), y_np, 'r-', label='target (cos)')
plt.plot(np.arange(x_np.size), x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 3)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = self.out(r_out[:, -1, :])
        return outs, h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state

        # or even simpler, since nn.Linear can accept inputs of any dimension
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
# loss_func = nn.MSELoss()
loss_func = nn.CrossEntropyLoss()
h_state = None  # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()  # continuously plot
pred_y_list = []
pred_y_list2 = []
total_ac = []
for step in range(1900):
    start, end = step, step + TIME_STEP  # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=int,
                        endpoint=False)

    x_np_train = x_np[steps]
    y_np_train = y_np[steps]

    x = torch.from_numpy(x_np_train[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np_train[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)  # rnn output
    # !! next step is important !!
    h_state = h_state.data  # repack the hidden state, break the connection from last iteration
    loss = loss_func(prediction, y[-1, -1, :])  # calculate loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    # plotting
    # plt.plot(steps, y_np_train.flatten(), 'r-')
    # plt.plot(steps[-1], prediction.data.numpy().flatten(), 'b-')
    # plt.draw()
    # plt.pause(0.05)

    if step > 1800:
        x_np_test = x_np[steps + 1]
        y_np_test = y_np[steps + 1]

        x = torch.from_numpy(x_np_train[np.newaxis, :, np.newaxis])
        y = torch.from_numpy(y_np_train[np.newaxis, :, np.newaxis])

        test_output, h_state_temp = rnn(x, h_state)  # (samples, time_step, input_size)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        accuracy = float((pred_y == y[-1, -1, :].numpy()).astype(int).sum())
        print('Epoch: ', end + 1, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

        total_ac.append(accuracy)
        pred_y_list.append(pred_y[-1])
        pred_y_list2.append(y[-1, -1, :].numpy()[-1])


print('test accuracy: %.2f' % (sum(total_ac) / len(total_ac)))
plt.figure()
plt.plot(range(len(pred_y_list2)), pred_y_list2)
plt.plot(range(len(pred_y_list2)), pred_y_list)
plt.savefig("filename.png")
plt.show()
