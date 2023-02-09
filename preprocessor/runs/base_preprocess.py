# Based on https://github.com/NATSpeech/NATSpeech
import utils.commons.single_thread_env  # NOQA

import importlib

from utils.commons.hparams import hparams, set_hparams


def preprocess():
    assert hparams["preprocess_cls"] != ""

    pkg = ".".join(hparams["preprocess_cls"].split(".")[:-1])
    cls_name = hparams["preprocess_cls"].split(".")[-1]
    process_cls = getattr(importlib.import_module(pkg), cls_name)
    process_cls().process()


if __name__ == "__main__":
    set_hparams()
    preprocess()
