import torch


class LSTMnet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.hidden = None

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first=True  # because the input shape'll be (batch_size, seq_length, feature)
        )
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        #print('initial input (and to the lstm layer) shape: ', x.shape)

        if self.hidden == None:
            x, self.hidden = self.lstm(x)
        else:
            x, self.hidden = self.lstm(x, self.hidden)
        #print('output of lstm layer (and input to fc) shape: ', x.shape)

        x = self.fc(x)
        #print('output of fc layer (and final output) shape: ', x.shape)
        #print('test shape: ', x[:,-1,:].shape)

        return x[:, -1, :]