import Arguments as Args
import torch
import random
import Network

torch.manual_seed(1)

# LOAD DATA PART
print("\nLoading Data...")
import dataProcess.DataProcess as D
input_lang, output_lang, pairs = D.prepareData('eng', 'fra', False)

# Train
import Train_KFold as T
import Evaluate as E
from sklearn.model_selection import KFold
trainSize = Args.args.train_size
training_pairs = [D.variablesFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(trainSize)] # check if repetition exists
kf = KFold(n_splits=Args.args.kfold)
kf.get_n_splits(training_pairs)

k=0
for train_index, test_index in kf.split(training_pairs) :
    k = k + 1
    EncNet = Network.EncoderRNN(input_lang.n_words, Args.args.hidden_size)
    DecNet = Network.DecoderRNN(output_lang.n_words, Args.args.hidden_size)
    if Args.args.no_gpu == False:
        EncNet.cuda()
        DecNet.cuda()

    print("\nTraining...")
    for epoch in range(Args.args.epoch) :
        print("[%d]-Fold, epoch : [%d]" % (k, epoch))
        T.TrainIters(train_index, training_pairs, EncNet, DecNet, trainSize=trainSize, epoch_size=Args.args.epoch, batch_size=Args.args.batch_size, lr=Args.args.learning_rate)
    print("\nDone Training !")

    print("Evaluation at [%d]-Fold" % k)
    E.Evaluate(EncNet, DecNet, test_index, output_lang, training_pairs)


# Print Result with arguments both into prompt and file
filename = "test"
outfile = open("%s.txt" % filename,"w")
print("\nParameters :")
for attr, value in sorted(Args.args.__dict__.items()) :
    print("\t{}={}".format(attr.upper(), value))
    outfile.write("\t{}={}\n".format(attr.upper(), value))
