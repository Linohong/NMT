import Arguments as Args

# LOAD DATA PART
print("\nLoading Data...")
import dataProcess.DataProcess as D
input_lang, output_lang, pairs = D.prepareData('eng', 'fra', True)

# CALL MODEL
import Network
EncNet = Network.EncoderRNN(input_lang.n_words, Args.args.hidden_size)
DecNet = Network.DecoderRNN(output_lang.n_words, Args.args.hidden_size)
if Args.args.no_gpu == False :
    EncNet.cuda()
    DecNet.cuda()

# Train
print("\nTraining...")
import Train as T
T.TrainIters(input_lang, output_lang, EncNet, DecNet, pairs, trainSize=Args.args.train_size, epoch_size=Args.args.epoch, batch_size=Args.args.batch_size, lr=Args.args.learning_rate)
print("\nDone Training !")


# Print Result with arguments both into prompt and file
filename = "test"
outfile = open("%s.txt" % filename,"w")
print("\nParameters :")
for attr, value in sorted(Args.args.__dict__.items()) :
    print("\t{}={}".format(attr.upper(), value))
    outfile.write("\t{}={}\n".format(attr.upper(), value))
