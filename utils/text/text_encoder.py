# Based on https://github.com/NATSpeech/NATSpeech
import json
import re
import six

from six.moves import range

from utils.text.ko_symbols import symbols


PAD = "<pad>"
EOS = "<EOS>"
UNK = "<UNK>"
SEG = "|"
PUNCS = "!./?;:"
RESERVED_TOKENS = [PAD, EOS, UNK, SEG]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1
UNK_ID = RESERVED_TOKENS.index(UNK)  # Normally 2

if six.PY2:
    RESERVED_TOKENS_BYTES = RESERVED_TOKENS
else:
    RESERVED_TOKENS_BYTES = [bytes(PAD, "ascii"), bytes(EOS, "ascii")]


def strip_ids(ids: list, ids_to_strip: list):
    """ Strip ids_to_strip from the end ids. """
    ids = list@staticmethod
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


class TextEncoder(object):
    """ Base class for converting from ints to/from human readable strings."""
    
    def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
        self._num_reserved_ids = num_reserved_ids
    
    @property
    def num_reserved_ids(self):
        return self._num_reserved_ids
    
    def encode(self, s: str):
        """ Transform a human-readable string into a seqeunce of int ids.
        
        The ids should be in the range [num_reserved_ids, vocab_size). Ids [0,
        num_reserved_ids) are reserved.
        
        EOS is not appended.
        
        Parameters
        ----------
        s: str
            human-readable string to be converted.
        
        Returns
        -------
        ids: list
            list of integers
        """
        return [int(w) + self._num_reserved_ids for w in s.split()]
    
    def decode(self, ids: list, strip_extraneous=False):
        """ Transform a sequence of int ids into a their string versions.
        
        This method supports transforming individual input/output ids to their
        string versions so that sequence to/from text conversions can be visualized
        in a human-readable format.
        
        Parameters
        ----------
        ids: list
            list of integers of be converted.
        strip_extraneous: bool
            whether to stipr off extraneous tokens (BOS and PAD).
        
        Returns
        -------
        strs: str
            human-readable string.
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids: list):
        """ Transform a sequence of int ids into a their string versions.
        
        This method supports transforming individual input/output ids to thier
        string versions so that sequence to/from text conversisons can be visualized
        in a human-redable format.
        
        Parameters
        ----------
        ids: list
            list of integers to be converted.
            
        Returns
        -------
        strs: list
            list of human-readable string.
        """
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self.num_reserved_ids)
        
        return [str(d) for d in decoded_ids]

    @property
    def vocab_size(self):
        raise NotImplementedError()


class TokenTextEncoder(TextEncoder):
    """ Encoder based on a user-supplied vocabulary (file or list). """

    def __init__(self,
                 vocab_filename: str,
                 reverse=False,
                 vocab_list=None,
                 replace_oov=None,
                 num_reserved_ids=NUM_RESERVED_TOKENS):
        """ Initialize from a file or list, one token per line.
        
        Handling of reserved tokens works as follows:
         - When initializing from a list, we add reserved tokens to the vocab.
         - When initializing from a file, we do not add rserved tokens to the vocab.
         - When saving vocab files, we save reserved tokens to the file.
        
        Parameters
        ----------
        vocab_filename: str
            If not None, the full filename to read vocab from. If this is not None,
            then vocab_list should be None.
        reverse: bool
            Indicating if tokens sholud be reversed during encoding and decoding.
        vocab_list: list
            If not None, a list of elements of the vocabulary. If this is not None,
            then vocab_filename should be None.
        replace_oov: str
            If not None, every out-of-vocabulary token seen when encoding will be
            replaced by this string (which must be in vocab).
        num_reserved_ids: int
            Number of IDs to save for reserved tokens like <EOS>.
        """
        super(TokenTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        if vocab_filename:
            self._init_vocab_from_file(vocab_filename)
        else:
            assert vocab_list is not None
            self._init_vocab_from_list(vocab_list)
        self.pad_index = self.token_to_id[PAD]
        self.eos_index = self.token_to_id[EOS]
        self.unk_index = self.token_to_id[UNK]
        self.seg_index = self.token_to_id[SEG] if SEG in self.token_to_id else self.eos_index

    def encode(self, s: str):
        """ Converts a space-separated string of tokens to a list of ids. """
        sentence = s
        tokens = sentence.strip().split()
        if self._replace_oov is not None:
            tokens = [t if t in self.token_to_id else self._replace_oov
                        for t in tokens]
        ret = [self.token_to_id[tok] for tok in tokens]

        return ret[::-1] if self._reverse else ret

    def decode(self, ids: list, strip_eos=False, strip_padding=False):
        if strip_padding and self.pad() in list(ids):
            pad_pos = list(ids).index(self.pad())
            ids = ids[:pad_pos]
        if strip_eos and self.eos() in list(ids):
            eos_pos = list(ids).index(self.eos())
            ids = ids[:eos_pos]

        return " ".join(self.decode_list(ids))

    def decode_list(self, ids: list):
        seq = reversed(ids) if self._reverse else ids
        
        return [self._safe_id_to_token(i) for i in seq]
    
    @property
    def vocab_size(self):
        return len(self.id_to_token)
    
    def __len__(self):
        return self.vocab_size
    
    def _safe_id_to_token(self, idx: list):
        return self.id_to_token.get(idx, "ID_%d" % idx)

    def _init_vocab_from_file(self, filename: str):
        """ Load vocab from a file.
        
        Parameters
        ----------
        filename: str
            The file to load vocabulary from.
        """
        with open(filename) as f:
            tokens = [token.strip() for token in f.readlines()]

        def token_gen():
            for token in tokens:
                yield token
        
        self._init_vocab(token_gen(), add_reserved_tokens=False)

    def _init_vocab_from_list(self, vocab_list: list):
        """ Initialize tokens from a list of tokens.
        
        It is ok if reserved tokens appear in the vocab list. They will be
        removed. The set of tokens in vocab_list should be unique.
        
        Parameters
        ----------
        vocab_list: list
            A list of tokens
        """
        def token_gen():
            for token in vocab_list:
                if token not in RESERVED_TOKENS:
                    yield token
        
        self._init_vocab(token_gen())

    def _init_vocab(self, token_generator, add_reserved_tokens=True):
        """ Initialize vocabulary with tokens from token_generator. """

        self.id_to_token = {}
        non_reserved_start_index = 0

        if add_reserved_tokens:
            self.id_to_token.update(enumerate(RESERVED_TOKENS))
            non_reserved_start_index = len(RESERVED_TOKENS)

        self.id_to_token.update(
            enumerate(token_generator, start=non_reserved_start_index))

        # _token_to_id is the reverse of _id_to_token
        self.token_to_id = dict((v, k) for k, v in six.iteritems(self.id_to_token))

    def pad(self):
        return self.pad_index
    
    def eos(self):
        return self.eos_index

    def unk(self):
        return self.unk_index

    def seg(self):
        return self.seg_index

    def store_to_file(self, filename: str):
        """ Write vocab file to disk.
        
        Vocab files have one token per lien. The file ends in a newline. Reserved
        tokens are written to the vocab file as well.
        
        Parameters
        ----------
        filename: str
            Full path of the file to store the vocab to.
        """
        with open(filename, "w") as f:
            for i in range(len(self.id_to_token)):
                f.write(self.id_to_token[i] + "\n")

    def sil_phonemes(self):
        return [p for p in self.id_to_token.values() if is_sil_phoneme(p)]


def build_token_encoder(token_list_file: str):
    """
    Parameters
    ----------
    token_list_file: str
        Path of phoneme_set file
    """
    token_list = json.load(open(token_list_file))

    return TokenTextEncoder(None, vocab_list=token_list, replace_oov="<UNK>")


def is_sil_phoneme(p: str):
    return p == "" or not (p[0].isalpha() or ishangul(p[0]))


def ishangul(phoneme):
    if phoneme not in symbols:
        # Consider Hangul syllable
        hanCount = len(re.findall(u"[\u3130-\u318F\uAC00-\uD7A3]+", phoneme))
        return hanCount > 0
    else:
        # Consider Hanguel jamo
        return phoneme in symbols
