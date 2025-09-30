import torch.nn as nn


class RNN(nn.Module):
    """A residual recurrent neural network.

    The input has shape (B, L, F), where B is the batch size, L is the sequence
    length, and F is the dimension of a single input vector, i.e. the number of
    aggregated time series.

    This class assumes the series we are interested in predicting is the first
    position of the third dimension: F = 0. This is important because only that
    one position is sent through the skip connections.

    The rest of the implementation is standard.
    """

    def __init__(
        self,
        seq_length,
        input_dim,
        n_layers,
        hidden_dim,
        output_dim=1,
        nonlinearity="tanh",
        dropout=0.1,
    ):
        super(RNN, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            nonlinearity=nonlinearity,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x) # (B, L, F)
        h_last = rnn_out[:, -1, :] # (B, F)
        delta = self.linear(h_last) # (B, 1)
        last_value = x[:, -1, 0:1] # (B, 1)
        y_hat = last_value + delta # (B, 1) 
        return y_hat


class GRU(nn.Module):
    """A residual gate recurrent network.
    
    Exactly the same implementation as the RNN, except for using
    GRU instead of RNN. 

    TODO: Create a base class for both and only assign the specific 
    state function (RNN or GRU) in the child classes.s
    """

    def __init__(
        self,
        seq_length,
        input_dim,
        n_layers,
        hidden_dim,
        output_dim=1,
        dropout=0.2,
    ):
        super(GRU, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x) # (B, L, F)
        h_last = gru_out[:, -1, :] # (B, F)
        delta = self.linear(h_last) # (B, 1)
        last_value = x[:, -1, 0:1] # (B, 1)
        y_hat = last_value + delta # (B, 1)
        return y_hat
