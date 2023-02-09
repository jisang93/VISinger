# Based on https://github.com/keithito/tacotron
# Defines the set of symbols used in text input to the model.

_JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
_JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
_JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

_VALID_CHARS = _JAMO_LEADS + _JAMO_VOWELS + _JAMO_TAILS
symbols = list(_VALID_CHARS)
