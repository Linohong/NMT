import Arguments as Args

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p) : # boolean return
    return len(p[0].split(' ')) < Args.args.max_sent and \
        len(p[1].split(' ')) < Args.args.max_sent and \
        p[0].startswith(eng_prefixes)

def filterPairs(pairs) :
    return [pair for pair in pairs if filterPair(pair)]
