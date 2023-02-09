# Based on https://github.com/NATSpeech/NATSpeech
import utils.commons.single_thread_env  # NOQA
import importlib

from utils.commons.hparams import hparams, set_hparams


def binarize():
    binarizer_cls = hparams.get("binarizer_cls", "preprocessor.base_binarizer.BaseBinarizer")
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    print("| Binarizer: ", binarizer_cls)
    binarizer_cls().process()


if __name__ == "__main__":
    set_hparams()
    binarize()
