import argparse

# ARGUMENT PART
parser = argparse.ArgumentParser(description='Seq2Seq NMT')
# option
parser.add_argument('-no_gpu', type=bool, default=False, help='disable the gpu')
# model
parser.add_argument('-hidden_size', type=int, default=300)
parser.add_argument('-max_sent', type=int, default=10, help='max sentence length')
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-embed_dim', type=int, default=300)
# learning
parser.add_argument('-epoch', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=1, help='batch size for training [default: 64]')
parser.add_argument('-learning_rate', type=float, default=0.02, help='learning rate')
# Data
parser.add_argument('-train_size', type=int, default=70000, help='train size')
args = parser.parse_args()
