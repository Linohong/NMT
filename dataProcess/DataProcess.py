import readLangs as RL
import filterPair as FP
import torch
import Arguments as Args
from torch.autograd import Variable
SOS_token = 0
EOS_token = 1

def prepareData(lang1, lang2, reverse=False) :
    input_lang, output_lang, pairs = RL.readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = FP.filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    if Args.args.train_size > len(pairs) :
        Args.args.train_size = len(pairs)
    print("Counting Words...")
    for pair in pairs :
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted Words : ")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence (lang, sentence) :
    # it returns indexes of words of 'sentence' in 'lang' Lang object
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence) :
    # from given indexes of words that are the components of 'sentence',
    # i) append 'End of Sentence' mark at the end of the indexes
    # ii) wraps the indexes with Tensor and than with Variable.
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if ( Args.args.no_gpu ) :
        return result
    else :
        return result.cuda()

def variablesFromPair(input_lang, output_lang, pair) :
    # take pair of languages,
    # then call function which returns Variable of indexes.
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


