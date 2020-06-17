# it is a simple example for LSTM to predict time series
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# read data
flight_data = sns.load_dataset("flights")  # download data from GitHub and save in seaborn-data

# plot time series
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'])

# pre-process
# transform data into float type
all_data = flight_data['passengers'].values.astype(float)
# split train and test data
test_data_size = 12
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
# normalize data into [-1,1]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
# transform by train window
train_window = 12
train_inout_seq = []
L = len(train_data_normalized)
for i in range(L - train_window):
    train_seq = train_data_normalized[i:i + train_window]
    train_label = train_data_normalized[i + train_window:i + train_window + 1]
    train_inout_seq.append((train_seq, train_label))


# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        # hidden state and cell state
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # input: [input_seq] => [input_seq, 1, 1]
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, 1), self.hidden_cell)
        # out: [input_seq, batch, hidden]
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # the last value of out is used to estimate the next value
        return predictions[-1]


# build model
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train
epochs = 150
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# prediction
fut_pred = 12
test_inputs = train_data_normalized[-train_window:].tolist()  # the last 12 data are used to predict test
model.eval()
for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

# transform into actual value without normalization
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))

# plot prediction
x = np.arange(132, 144, 1)
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'])
plt.plot(x, actual_predictions)
plt.show()
