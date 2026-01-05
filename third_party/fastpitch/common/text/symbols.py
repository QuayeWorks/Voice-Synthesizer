""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from .cmudict import valid_symbols


# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in valid_symbols]


STYLE_TAGS = [
    "<narration>",
    "<dialogue>",
    "<inner>",
    "<calm>",
    "<emphasis>",
    "<angry>",
    "<shout>",
    "<whisper>",
    "<battle>",
    "<comedy>",
]


def get_symbols(
    symbol_set="english_basic",
    include_style_tokens=True,
    extra_symbols=None,
    style_tags=None,
):
    extra_symbols = extra_symbols or []
    active_style_tags = style_tags or STYLE_TAGS
    style_tokens = active_style_tags if include_style_tokens else []
    extra_symbols = list(style_tokens) + [s for s in extra_symbols if s not in style_tokens]
    if symbol_set == 'english_basic':
        _pad = '_'
        _punctuation = '!\'(),.:;? '
        _special = '-'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_basic_lowercase':
        _pad = '_'
        _punctuation = '!\'"(),.:;? '
        _special = '-'
        _letters = 'abcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_expanded':
        _punctuation = '!\'",.:;? '
        _math = '#%&*+-/[]()'
        _special = '_@©°½—₩€$'
        _accented = 'áçéêëñöøćž'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_punctuation + _math + _special + _accented + _letters) + _arpabet
    else:
        raise Exception("{} symbol set does not exist".format(symbol_set))

    for sym in extra_symbols:
        if sym not in symbols:
            symbols.append(sym)

    return symbols


def get_pad_idx(symbol_set='english_basic'):
    if symbol_set in {'english_basic', 'english_basic_lowercase'}:
        return 0
    else:
        raise Exception("{} symbol set not used yet".format(symbol_set))
