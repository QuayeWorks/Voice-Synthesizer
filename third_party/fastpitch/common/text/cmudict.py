""" from https://github.com/keithito/tacotron """

import importlib.resources
import importlib.util
import re
from pathlib import Path
from urllib.request import urlopen


valid_symbols = [
  'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

_valid_symbol_set = set(valid_symbols)

CMUDICT_URLS = [
  # Primary source maintained by CMU Sphinx
  "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict-0.7b",
  # Mirror that remains stable when the primary is unreachable
  "https://raw.githubusercontent.com/Alexir/CMUdict/master/cmudict-0.7b",
]


def _download_cmudict(target: Path):
  target = Path(target)
  target.parent.mkdir(parents=True, exist_ok=True)

  print(f"CMUdict missing at {target}. Attempting download…")

  errors = []
  for url in CMUDICT_URLS:
    try:
      with urlopen(url) as response:
        target.write_bytes(response.read())
      print(f"CMUdict download complete from {url}.")
      return
    except Exception as exc:  # pragma: no cover - defensive fallback
      errors.append(f"{url} ({exc})")

  # Fallback: use the PyPI cmudict package if it is already installed.
  spec = importlib.util.find_spec("cmudict")
  if spec is not None:
    resource_root = importlib.resources.files("cmudict")
    for resource_name in ("cmudict-0.7b", "cmudict.dict"):
      resource = resource_root / resource_name
      if resource.is_file():
        target.write_bytes(resource.read_bytes())
        print(f"CMUdict copied from installed 'cmudict' package ({resource_name}).")
        return
    errors.append("Installed 'cmudict' package does not contain a usable dictionary file.")
  else:
    errors.append("PyPI package 'cmudict' is not installed.")

  error_report = "\n  - ".join(errors)
  raise RuntimeError(f"Failed to obtain CMUdict. Tried:\n  - {error_report}")


def lines_to_list(filename):
  with open(filename, encoding='utf-8') as f:
    lines = f.readlines()
  lines = [l.rstrip() for l in lines]
  return lines


class CMUDict:
  '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''
  def __init__(self, file_or_path=None, heteronyms_path=None, keep_ambiguous=True):
    if file_or_path is None:
      self._entries = {}
    else:
      self.initialize(file_or_path, keep_ambiguous)

    if heteronyms_path is None:
      self.heteronyms = []
    else:
      self.heteronyms = set(lines_to_list(heteronyms_path))

  def initialize(self, file_or_path, keep_ambiguous=True):
    if isinstance(file_or_path, (str, Path)):
      file_or_path = Path(file_or_path)
      try:
        with open(file_or_path, encoding='latin-1') as f:
          entries = _parse_cmudict(f)
      except FileNotFoundError:
        try:
          _download_cmudict(file_or_path)
          with open(file_or_path, encoding='latin-1') as f:
            entries = _parse_cmudict(f)
        except Exception as exc:  # pragma: no cover - defensive fallback
          print("Failed to automatically download CMUdict.")
          print(f"Error: {exc}")
          print()
          print("You can manually download the CMU Pronouncing Dictionary from:")
          print(f"  {CMUDICT_URLS[0]}")
          print(f"and place it at: {file_or_path}")
          import sys
          sys.exit(1)
    else:
      entries = _parse_cmudict(file_or_path)
    if not keep_ambiguous:
      entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
    self._entries = entries

  def __len__(self):
    if len(self._entries) == 0:
      raise ValueError("CMUDict not initialized")
    return len(self._entries)

  def lookup(self, word):
    '''Returns list of ARPAbet pronunciations of the given word.'''
    if len(self._entries) == 0:
      raise ValueError("CMUDict not initialized")
    return self._entries.get(word.upper())


_alt_re = re.compile(r'\([0-9]+\)')


def _parse_cmudict(file):
  cmudict = {}
  for line in file:
    if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
      parts = line.split('  ')
      word = re.sub(_alt_re, '', parts[0])
      pronunciation = _get_pronunciation(parts[1])
      if pronunciation:
        if word in cmudict:
          cmudict[word].append(pronunciation)
        else:
          cmudict[word] = [pronunciation]
  return cmudict


def _get_pronunciation(s):
  parts = s.strip().split(' ')
  for part in parts:
    if part not in _valid_symbol_set:
      return None
  return ' '.join(parts)
