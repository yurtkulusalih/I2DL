import pickle
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .rnn_nn import *
from .base_classifier import *


class RNN_Classifier(Base_Classifier):
    
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()

        ############################################################################
        #  TODO: Build a RNN classifier                                            #
        ############################################################################
        self.classes = classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity=activation)
        self.FC_layer = nn.Linear(hidden_size, classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, x):
        
        ############################################################################
        #  TODO: Perform the forward pass                                          #
        ############################################################################   
        self.hidden = self.init_hidden(x.size(1))
        rnn_out, self.hidden = self.rnn(x, self.hidden)
        out = self.FC_layer(self.hidden)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return out.view(-1, self.classes)



class LSTM_Classifier(Base_Classifier):

    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        
        self.classes = classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.FC_layer = nn.Linear(hidden_size, classes)
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    
    def init_cell_state_and_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, x):

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################    
        self.hidden, self.cell_state = self.init_cell_state_and_hidden(x.size(1))
        lstm_out, (self.hidden, self.cell_state) = self.lstm(x, (self.hidden, self.cell_state))
        out = self.FC_layer(self.hidden)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return out.view(-1, self.classes)
