# Based on https://github.com/NATSpeech/NATSpeech
from utils.text.text_encoder import is_sil_phoneme

REGISTERED_TEXT_PROCESSORS = {}


def register_text_processors(name):
    # Regist text_processors to REGISTERED_TEXT_PROCESSORS
    def _f(cls):
        REGISTERED_TEXT_PROCESSORS[name] = cls
        return cls
    
    return _f


def get_text_processor_cls(name):
    # Get text_processors from REGSISTERED_TEXT_PROCESSORS
    return REGISTERED_TEXT_PROCESSORS.get(name, None)


class BaseTextProcessor:
    @staticmethod
    def sp_phonemes():
        return ['|']

    @classmethod
    def process(cls, text, preprocess_args):
        raise NotImplementedError
    
    @classmethod
    def postprocess(cls, text_struct, preprocess_args):
        # Remove sil_phoneme in head and tail
        while len(text_struct) > 0 and is_sil_phoneme(text_struct[0][0]):
            text_struct = text_struct[1:]
        while len(text_struct) > 0 and (is_sil_phoneme(text_struct[-1][0]) and text_struct[-1][0] not in cls.PUNCS):
            text_struct = text_struct[:-1]
        if preprocess_args["with_phsep"]:  # Add ph to each word
            text_struct = cls.add_bdr(text_struct)
        if preprocess_args["add_eos_bos"]:  # Add EOS and BOS token
            text_struct = [["<BOS>", ["<BOS>"]]] + text_struct + [["<EOS>", ["<EOS>"]]]
        return text_struct
    
    @classmethod
    def add_bdr(cls, text_struct):
        text_struct_ = []
        for i, ts in enumerate(text_struct):
            text_struct_.append(ts)
            if i != len(text_struct) - 1 and \
                    not is_sil_phoneme(text_struct[i][0]) and not is_sil_phoneme(text_struct[i + 1][0]):
                text_struct_.append(['|', ['|']])
        return text_struct_
