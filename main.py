import scipy.io as sio  # load .mat files (ts)
import numpy as np
import matplotlib.pyplot as plt
import torch
from LSTMnet import LSTMnet

# Loading ts (.mat files)
mat_content = sio.loadmat('/home/marzs8/efop/v.mat')

labels = []
for key in mat_content.keys():  # sio.loadmat() returns a dict with variable names as keys
    if key.find('Meresek') == -1:
        continue
    labels.append(key)

ts_data = np.empty((len(labels),  # this will store the ts data
                    mat_content[labels[0]].shape[0],
                    mat_content[labels[0]].shape[1]
                    ))

for idx, label in enumerate(labels):  # loading data into the np array
    ts_data[idx] = mat_content[label]

ts = ts_data[10, 0]

# plt.plot(ts[0:168])
# plt.show()

ts = np.expand_dims(ts, axis=1)

def create_sequences(ts, seq_len):
    xs = []
    ys = []

    for i in range(len(ts) - seq_len):
        x = ts[i:(i + seq_len)]
        y = ts[i + seq_len]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

seq_len = 24  # 7 x 24h = 168h (1w)
x_train, y_train = create_sequences(ts, seq_len)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()

print('x_train shape: ', x_train.shape,
      'y_train shape: ', y_train.shape)

# TODO: ezeket fent kiszervezni f"uggv'enyekbe

# print(torch.cuda.current_device()) -- nem megy a CUDA!


def train(model, num_epochs):
    lr = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    losses = []
    
    for epoch in range(num_epochs):
        y_pred = model(x_train)
        # print('y_pred shape: ', y_pred.shape)

        loss = loss_fn(y_pred.float(), y_train)
        losses.append(loss.item())
        model.hidden = None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1} / {num_epochs}] \t Loss {losses[-1]}')

    return losses, model


input_size = 1
hidden_size = 60
num_layers = 1

num_epochs = 1000

model = LSTMnet(input_size, hidden_size, num_layers)
losses, model = train(model, num_epochs)

plt.plot(losses)
plt.show()

with torch.no_grad():
    y_pred = model(x_train)
    plt.plot(y_train[0:200, 0])
    plt.plot(y_pred[0:200, 0])
    plt.show()






