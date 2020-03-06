# an example for RNN: learning sin function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

# parameters
num_time_steps = 50
input_size = 1
hidden_size = 16
output_size = 1
lr = 0.01

# network
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        # initialize w_ih, w_hh, b_ih, b_hh
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

        # linear connection
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [1, seq, h] => [seq, h]
        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        # [1, 49, 1] & [1, 1, 16]
        return out, hidden_prev


model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

# train
hidden_prev = torch.zeros(1, 1, hidden_size)
for iter in range(6000):
    # sample from {0,1,2}
    start = np.random.randint(3, size=1)[0]
    # split [start, start + 10] into num_time_steps pieces
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    # [1, 49, 1]
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

    output, hidden_prev = model(x, hidden_prev)
    # the value of hidden_prev
    hidden_prev = hidden_prev.detach()

    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print("Iteration: {} loss {}".format(iter, loss.item()))

# prediction
start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

predictions = []
# [1, 1]
input = x[:, 0, :]
for _ in range(x.shape[1]):
    # [1, 1] => [1, 1, 1]
    input = input.view(1, 1, 1)
    # one step ahead prediction
    pred, hidden_prev = model(input, hidden_prev)
    input = pred
    # transform into scalar
    predictions.append(pred.detach().numpy().ravel()[0])

# plot
x = x.data.numpy().ravel()
plt.scatter(time_steps[:-1], x, s=90)
plt.scatter(time_steps[1:], predictions)
plt.show()


