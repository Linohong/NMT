import Unicode2Ascii as U2A
from Lang import Lang
from io import open

def readLangs(lang1, lang2, reverse=False) :
    print("Reading Lines...")

    # Read the file and split into lines
    lines = open('../data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[U2A.normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instanes
    if reverse :
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else :
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

