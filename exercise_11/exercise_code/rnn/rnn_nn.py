import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.input_weight = nn.Linear(input_size, hidden_size)
        self.hidden_weight = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh() if activation == "tanh" else nn.ReLU()

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        
        if h is None:
            h = Variable(torch.zeros(1, x.size(1), self.hidden_size))
        seq_len = x.size(0)
        h_seq = Variable(torch.zeros(seq_len, x.size(1), self.hidden_size))
        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   
        for i in range(seq_len):
                h = self.activation(self.hidden_weight(h) + self.input_weight(x[i]))
                h_seq[i] = h.clone()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        """
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.Tanh_activation = nn.Tanh()
        self.Sigmoid_activation = nn.Sigmoid()
        
        self.forget_weight = nn.Linear(input_size, hidden_size)
        self.forget_hidden_weight = nn.Linear(hidden_size, hidden_size)
        
        self.input_weight = nn.Linear(input_size, hidden_size)
        self.input_hidden_weight = nn.Linear(hidden_size, hidden_size)
        
        self.output_weight = nn.Linear(input_size, hidden_size)
        self.output_hidden_weight = nn.Linear(hidden_size, hidden_size)
        
        self.cell_state_weight = nn.Linear(input_size, hidden_size)
        self.cell_state_hidden_weight = nn.Linear(hidden_size, hidden_size)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################       


    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        
        if h is None:
            h = Variable(torch.zeros(1, x.size(1), self.hidden_size))
        if c is None:
            c = Variable(torch.zeros(1, x.size(1), self.hidden_size))
        
        seq_len = x.size(0)
        h_seq_temp = []

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   
        for i in range(seq_len):
            X = x[i]
            f = self.Sigmoid_activation(self.forget_weight(X) + self.forget_hidden_weight(h))
            i = self.Sigmoid_activation(self.input_weight(X) + self.input_hidden_weight(h))
            o = self.Sigmoid_activation(self.output_weight(X) + self.output_hidden_weight(h))
            c = f * c + i * self.Tanh_activation(self.cell_state_weight(X) + self.cell_state_hidden_weight(h))
            h = o * self.Tanh_activation(c)
            h_seq_temp.append(h)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        h_seq = torch.zeros(seq_len, x.size(1), self.hidden_size)
        for idx in range(len(h_seq_temp)):
            h_seq[idx] = h_seq_temp[idx]
      
        return h_seq , (h, c)

