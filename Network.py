# Encoder Network
import torch
import torch.nn as nn
from torch.autograd import Variable
import Arguments as Args

class EncoderRNN(nn.Module) :
    def __init__ (self, vocab_size, hidden_size) :
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden) :
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self) :
        result = Variable(torch.zeros(1, 1, self.hidden_size)) # initialize with zeros as hidden_size.
        if not Args.args.no_gpu :
            return result.cuda()
        else :
            return result

class DecoderRNN(nn.Module) :
    def __init__ (self, vocab_size, hidden_size) :
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden) :
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden) # gru(input, h_0) : input=>(seq_len, batch, input_size)
        output = self.softmax(self.fc(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if not Args.args.no_gpu:
            return result.cuda()
        else:
            return result