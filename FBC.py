import scipy.io as sio  # load .mat files (ts)
import numpy as np
import torch  # PyTorch
from LSTMnet import LSTMnet


class FBC:
    def __init__(self, seq_len, M):
        self.seq_len = seq_len
        self.N = 12  # TODO: hardcoded
        self.M = M


        # Load data
        self.ts_data = self.load_data()

        # Data preprocessing
        self.ts = self.ts_data[:, 0:self.M]  # TODO: more sophisticated, check for mostly zero seqs

        self.x_trains, self.y_trains = self.create_sequences()

        # print(torch.cuda.current_device()) -- nem megy a CUDA! TODO: resolve CUDA

        # LSTMnet model parameters TODO: not hardcoded
        self.input_size = 1
        self.hidden_size = 60
        self.num_layers = 1

        self.num_epochs = 1

        # Creating the nets and training
        self.nets, self.losses = self.train()

    def load_data(self):  # TODO: very specific, hard-coded - generalize!
        # Loading ts (.mat files)

        # mat_content = sio.loadmat('/home/marzs8/efop/v.mat')
        mat_content = sio.loadmat("C:/Users/zsolt/Desktop/v.mat")

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

        return ts_data

    def create_sequences(self):
        x_trains = []
        y_trains = []

        for i in range(self.N):
            x_trains_temp_list = []
            y_trains_temp_list = []

            for j in range(self.M):
                xs = []
                ys = []

                for k in range(len(self.ts[i, j]) - self.seq_len):
                    x = self.ts[i, j, k:(k + self.seq_len)]
                    y = self.ts[i, j, k + self.seq_len]

                    xs.append(x)
                    ys.append(y)

                xs = np.expand_dims(np.array(xs), axis=2)
                ys = np.expand_dims(np.array(ys), axis=1)

                x_trains_temp_list.append(torch.from_numpy(xs).float())
                y_trains_temp_list.append(torch.from_numpy(ys).float())

            x_trains.append(x_trains_temp_list)
            y_trains.append(y_trains_temp_list)

        return x_trains, y_trains

    def train(self):
        # Creating the FBC nets array

        nets = []
        losses = []

        for i in range(self.N):
            temp_nets = []
            temp_losses = []

            for j in range(self.M):
                model = LSTMnet(self.input_size, self.hidden_size, self.num_layers)

                lr = 1e-3
                optimizer = torch.optim.Adam(model.parameters(), lr)
                loss_fn = torch.nn.MSELoss(reduction='sum')

                loss_list = []

                for epoch in range(self.num_epochs):
                    y_pred = model(self.x_trains[i][j])
                    # print('y_pred shape: ', y_pred.shape)

                    loss = loss_fn(y_pred.float(), self.y_trains[i][j])
                    loss_list.append(loss.item())
                    model.hidden = None

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(f'Net[{i},{j}] of {self.N} x {self.M}; Epoch [{epoch + 1} / {self.num_epochs}] \t Loss {loss_list[-1]}')

                temp_nets.append(model)
                temp_losses.append(loss_list)

            nets.append(temp_nets)
            losses.append(temp_losses)

        return nets, losses
