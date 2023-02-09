# Based on https://github.com/NATSpeech/NATSpeech
import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import importlib

from utils.commons.hparams import hparams, set_hparams


def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()


if __name__ == '__main__':
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(i) for i in range(torch.cuda.device_count())])
    set_hparams()
    run_task()

