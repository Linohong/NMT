# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427

import unicodedata
import re

def unicodeToAscii(s) :
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-Letter characters
def normalizeString(s) :
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s