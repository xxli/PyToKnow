import re
import unicodedata

def is_digit(s):
    '''Determine whether the str matches a integer or a float number.
    '''
    value = re.compile(r'^[-+]?\d*\.{0,1}\d+$')
    result=value.match(s)
    if result:
        return True
    else:
        return False

def unicode2ascii(s):
    '''Turn a Unicode string to plain ASCII, thanks to
    https://stackoverflow.com/a/518232/2809427
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    '''Lowercase, trim, and remove non-letter characters.
    '''
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s