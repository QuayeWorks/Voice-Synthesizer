""" adapted from https://github.com/keithito/tacotron """
import re
import numpy as np
from . import cleaners
from .symbols import STYLE_TAGS, get_symbols
from . import cmudict
from .numerical import _currency_re, _expand_currency


#########
# REGEX #
#########

# Regular expression matching text enclosed in curly braces for encoding
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

# Regular expression matching words and not words
_words_re = re.compile(r"([a-zA-ZÀ-ž]+['][a-zA-ZÀ-ž]{1,2}|[a-zA-ZÀ-ž]+)|([{][^}]+[}]|[^a-zA-ZÀ-ž{}]+)")

# Regular expression separating words enclosed in curly braces for cleaning
_arpa_re = re.compile(r'{[^}]+}|\S+')


class TextProcessing(object):
    def __init__(self, symbol_set, cleaner_names, p_arpabet=0.0,
                 handle_arpabet='word', handle_arpabet_ambiguous='ignore',
                 expand_currency=True, include_style_tokens=True,
                 style_tags=None, strip_style_from_text=False):
        self.include_style_tokens = include_style_tokens
        self.style_tags = style_tags or STYLE_TAGS
        self.strip_style_from_text = strip_style_from_text or not include_style_tokens

        extra_symbols = self.style_tags if include_style_tokens else None
        self.symbols = get_symbols(symbol_set, include_style_tokens=include_style_tokens,
                                   extra_symbols=extra_symbols)
        self.cleaner_names = cleaner_names

        # Mappings from symbol to numeric ID and vice versa:
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}
        self.expand_currency = expand_currency

        # cmudict
        self.p_arpabet = p_arpabet
        self.handle_arpabet = handle_arpabet
        self.handle_arpabet_ambiguous = handle_arpabet_ambiguous


    def text_to_sequence(self, text):
        sequence = []

        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self.symbols_to_sequence(text)
                break
            sequence += self.symbols_to_sequence(m.group(1))
            sequence += self.arpabet_to_sequence(m.group(2))
            text = m.group(3)

        return sequence

    def sequence_to_text(self, sequence):
        result = ''
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol:
                s = self.id_to_symbol[symbol_id]
                # Enclose ARPAbet back in curly braces:
                if len(s) > 1 and s[0] == '@':
                    s = '{%s}' % s[1:]
                result += s
        return result.replace('}{', ' ')

    def clean_text(self, text):
        for name in self.cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception('Unknown cleaner: %s' % name)
            text = cleaner(text)

        return text

    def symbols_to_sequence(self, symbols):
        return [self.symbol_to_id[s] for s in symbols if s in self.symbol_to_id]

    def arpabet_to_sequence(self, text):
        return self.symbols_to_sequence(['@' + s for s in text.split()])

    def get_arpabet(self, word):
        arpabet_suffix = ''

        if word.lower() in cmudict.heteronyms:
            return word

        if len(word) > 2 and word.endswith("'s"):
            arpabet = cmudict.lookup(word)
            if arpabet is None:
                arpabet = self.get_arpabet(word[:-2])
                arpabet_suffix = ' Z'
        elif len(word) > 1 and word.endswith("s"):
            arpabet = cmudict.lookup(word)
            if arpabet is None:
                arpabet = self.get_arpabet(word[:-1])
                arpabet_suffix = ' Z'
        else:
            arpabet = cmudict.lookup(word)

        if arpabet is None:
            return word
        elif arpabet[0] == '{':
            arpabet = [arpabet[1:-1]]

        # XXX arpabet might not be a list here
        if type(arpabet) is not list:
            return word

        if len(arpabet) > 1:
            if self.handle_arpabet_ambiguous == 'first':
                arpabet = arpabet[0]
            elif self.handle_arpabet_ambiguous == 'random':
                arpabet = np.random.choice(arpabet)
            elif self.handle_arpabet_ambiguous == 'ignore':
                return word
        else:
            arpabet = arpabet[0]

        arpabet = "{" + arpabet + arpabet_suffix + "}"

        return arpabet

    def encode_text(self, text, return_all=False, return_style_id=False):
        style_tag, text = self._extract_style_tag(text)
        style_id = None

        if style_tag is not None and self.include_style_tokens:
            style_id = self.symbol_to_id.get(style_tag)

        text_encoded, text_clean, text_arpabet = self._encode_main_text(
            text, return_all=True)

        sequence = []
        if style_tag is not None and self.include_style_tokens and not self.strip_style_from_text:
            if style_id is None:
                raise KeyError(f"Style tag {style_tag} missing from symbols list")
            sequence.append(style_id)
            text_clean = f"{style_tag} {text_clean}".strip()
            if text_arpabet:
                text_arpabet = f"{style_tag} {text_arpabet}".strip()

        sequence.extend(text_encoded)

        if return_all or return_style_id:
            parts = [sequence]
            if return_all:
                parts.extend([text_clean, text_arpabet])
            if return_style_id:
                parts.append(style_id)
            return tuple(parts)

        return sequence

    def _encode_main_text(self, text, return_all=False):
        if self.expand_currency:
            text = re.sub(_currency_re, _expand_currency, text)
        text_clean = [self.clean_text(split) if split[0] != '{' else split
                      for split in _arpa_re.findall(text)]
        text_clean = ' '.join(text_clean)
        text_clean = cleaners.collapse_whitespace(text_clean)
        text = text_clean

        text_arpabet = ''
        if self.p_arpabet > 0:
            if self.handle_arpabet == 'sentence':
                if np.random.uniform() < self.p_arpabet:
                    words = _words_re.findall(text)
                    text_arpabet = [
                        self.get_arpabet(word[0])
                        if (word[0] != '') else word[1]
                        for word in words]
                    text_arpabet = ''.join(text_arpabet)
                    text = text_arpabet
            elif self.handle_arpabet == 'word':
                words = _words_re.findall(text)
                text_arpabet = [
                    word[1] if word[0] == '' else (
                        self.get_arpabet(word[0])
                        if np.random.uniform() < self.p_arpabet
                        else word[0])
                    for word in words]
                text_arpabet = ''.join(text_arpabet)
                text = text_arpabet
            elif self.handle_arpabet != '':
                raise Exception("{} handle_arpabet is not supported".format(
                    self.handle_arpabet))

        text_encoded = self.text_to_sequence(text)

        if return_all:
            return text_encoded, text_clean, text_arpabet

        return text_encoded

    def _extract_style_tag(self, text):
        if self.style_tags is None:
            return None, text

        trimmed = text.lstrip()
        for tag in self.style_tags:
            if trimmed.startswith(tag):
                trimmed = trimmed[len(tag):].lstrip()
                return tag, trimmed
        return None, text
