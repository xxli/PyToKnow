import unicodedata


def str_to_float(s):
    """字符串转换为float"""
    if s is None:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def precent_str_to_float(s):
    return float(s.rstrip('%')) / 100  # s='12%'


def chinese_char_to_float(s):
    return unicodedata.numeric(s)  # unicodedata.numeric('三'), unicodedata.numeric('二十一')
