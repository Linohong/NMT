import torch
from torch.autograd import Variable
import torch.optim as optim
import Network
import torch.nn as nn
import time
import random
import dataProcess.DataProcess as D
import peripheralTools as PT
import Arguments as Args


def Train(input_sent, target_sent, EncNet, DecNet, enc_optim, dec_optim, criterion, max_length=Args.args.max_sent) :
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    loss = 0
    input_length = input_sent.size()[0]
    target_length = target_sent.size()[0]

    # Encoder Part #
    enc_hidden = EncNet.initHidden() # initialized hidden Variable.
    enc_outputs = Variable(torch.zeros(max_length, EncNet.hidden_size)) # zeros of max_length * EncNet
    enc_outputs = enc_outputs if Args.args.no_gpu else enc_outputs.cuda()

    for ei in range(input_length) :
        enc_output, enc_hidden = EncNet(input_sent[ei], enc_hidden)
        enc_outputs[ei] = enc_output[0][0]

    # Decoder Part #
    dec_hidden = enc_hidden # initialize decoder's hidden state as enc_hidden state
    dec_input = Variable(torch.LongTensor([[D.SOS_token]])) # start of the input with SOS_token
    dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

    for di in range(target_length) :
        dec_output, dec_hidden = DecNet(dec_input, dec_hidden)
        topv, topi = dec_output.data.topk(1) # topk returns a tuple of (value, index)
        ni = topi[0][0] # next input

        dec_input = Variable(torch.LongTensor([[ni]]))
        dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

        loss += criterion(dec_output, target_sent[di])
        if ni == D.EOS_token :
            break

    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss.data[0] / target_length

def TrainIters(train_index, training_pairs, EncNet, DecNet, trainSize, print_every=1000, epoch_size=10, batch_size=50, lr=0.02) :
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(EncNet.parameters(), lr=lr)
    decoder_optimizer = optim.SGD(DecNet.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    iter_time = 1
    for iter in train_index :
        training_pair = training_pairs[iter]
        input_sent_variable = training_pair[0] # Variable of indexes of input sentence
        target_sent_variable = training_pair[1] # Variable of indexes of target sentence

        loss = Train(input_sent_variable, target_sent_variable, EncNet, DecNet, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        iter_time = iter_time + 1

    print_loss_avg = print_loss_total/trainSize
    print('%s (%d itered) %.4f' % (PT.timeSince(start, float(iter_time)/trainSize), iter_time, print_loss_avg ))


