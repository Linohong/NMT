import Arguments as Args
import torch
import dataProcess.DataProcess as D
from torch.autograd import Variable
import nltk

def Evaluate(EncNet, DecNet, test_index, output_lang, training_pairs, max_sent=Args.args.max_sent) :
    list_of_hypotheses = []
    list_of_references = []
    for ti in range(len(test_index)) :
        input_sent_variable = training_pairs[ti][0]
        output_sent_variable = training_pairs[ti][1]
        input_length = input_sent_variable.size()[0]
        output_length = output_sent_variable.size()[0]
        enc_hidden = EncNet.initHidden()

        enc_outputs = Variable(torch.zeros(max_sent, EncNet.hidden_size))
        enc_outputs = enc_outputs.cuda() if Args.args.no_gpu == False else enc_outputs

        for ei in range(input_length) :
            enc_output, enc_hidden = EncNet(input_sent_variable[ei], enc_hidden)
            enc_outputs[ei] = enc_outputs[ei] + enc_output[0][0] # why 2-dimensional?

        dec_input = Variable(torch.LongTensor([[D.SOS_token]]))
        dec_input = dec_input.cuda() if Args.args.no_gpu == False else dec_input
        dec_hidden = enc_hidden

        decoded_words = []
        for di in range(max_sent) :
            dec_output, dec_hidden = DecNet(dec_input, dec_hidden)
            topv, topi = dec_output.data.topk(1)
            ni = topi[0][0]
            if ni == D.EOS_token :
                decoded_words.append('<EOS>')
                break
            else :
                decoded_words.append(output_lang.index2word[ni])

            dec_input = Variable(torch.LongTensor([[ni]]))
            dec_input = dec_input.cuda() if Args.args.no_gpu == False else dec_input

        reference = []
        references = []
        for di in output_sent_variable.data :
            reference.append(output_lang.index2word[int(di)])
        references.append(reference)
        list_of_references.append(references)
        list_of_hypotheses.append(decoded_words)

    print('BLEU SCORE : %.5f\n\n', nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses))



