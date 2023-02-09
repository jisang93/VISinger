# Based on https://github.com/NATSpeech/NATSpeech
import os
import numpy as np
import sys
import traceback
import types
import torch
import torchaudio

from functools import wraps
from itertools import chain
from torch.utils.data import ConcatDataset, Dataset

from utils.commons.hparams import hparams


def collate_1d_or_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d(values, pad_idx, left_pad, shift_right, max_len, shift_id)
    else:
        return collate_2d(values, pad_idx, left_pad, shift_right, max_len)


def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(indices, num_tokens_fn, max_tokens=None, max_sentences=None,
                  required_batch_size_multiple=1, distributed=False):
    """  Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Parameters
    ----------
    indices: List[int]
        ordered list of dataset indices
    num_tokens_fn: callable
        function that returns the number of tokens at a given index
    max_tokens: int, optional
        max number of tokens in each batch (default: None).
    max_sentences: int, optional
        max number of sentences in each batch (default: None).
    required_batch_size_multiple: int, optional
        require batch size to be a multiple of N (default: 1).
    """
    max_tokens = max_tokens if max_tokens is not None else sys.maxsize
    max_sentences = max_sentences if max_sentences is not None else sys.maxsize
    batch_size_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert sample_len <= max_tokens, (
            f"sentence at index {idx} of size {sample_len} exceeds max_tokens limit of {max_tokens}!")
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                batch_size_mult * (len(batch) // batch_size_mult), len(batch) % batch_size_mult)
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


def data_loader(fn):
    """
    Decorator to make any fix with this use the lazy property

    Parameters
    ----------
    fn: callable
        function for data loader
    """
    wraps(fn)  # Update function inofrmation
    attr_name = "_lazy_" + fn.__name__

    def _get_data_loader(self):
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                # Lazy evaluation, done only once.
                value = fn(self)
            except AttributeError as e:
                # Guard against AttributeError suppression.
                traceback.print_exc()
                error = f"{fn.__name__}: An AttributeError was encoutered: {str(e)}"
                raise RuntimeError(error) from e
            # Memorize evaluation.
            setattr(self, attr_name, value)
        return value
    
    return _get_data_loader


class BaseDataset(Dataset):
    def __init__(self, shuffle):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.sort_by_len = hparams["sort_by_len"]
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes
    
    def __getitem__(self, index):
        raise NotImplemented
    
    def collater(self, samples):
        raise NotImplementedError
    
    def __len__(self):
        return len(self._sizes)
    
    def num_tokens(self, index):
        return self.size(index)
    
    def size(self, index):
        """ Return an example's size as a float or tuple, This value is used
            when filtering a dataset with ``--max-positions``. """
        return min(self._sizes[index], self.hparams["max_tokens"])
    
    def ordered_indices(self):
        """ Return an ordered list of indices. Batches will be constructed
            based on this order. """
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind="mergesort")]
        else:
            indices = np.arange(len(self))
    
        return indices
    
    @property
    def num_workers(self):
        return int(os.getenv("NUM_WORKERS", self.hparams["ds_workers"]))
    
    def load_audio_to_torch(self, audio_path):
        """ [WARN] You have to normalize waveform by max wav value. """
        wav, sample_rate = torchaudio.load(audio_path, format="wav", normalize=False)
        # To ensure upsampling/downsampling will be processed in a right way for full signals
        return wav.squeeze(0), sample_rate


class BaseConcatDataset(ConcatDataset):
    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def _sizes(self):
        if not hasattr(self, "sizes"):
            self.sizes = list(chain.from_iterable([d._sizes for d in self.datasets]))
        return self.sizes
    
    def size(self, index):
        return min(self._sizes[index], hparams["max_frames"])
    
    def num_tokens(self, index):
        return self.size(index)
    
    def ordered_indices(self):
        """ Return an ordered list of indices. Batches will be constructed
            based on this order. """
        if self.datasets[0].shuffle:
            indices = np.random.permutation(len(self))
            if self.datasets[0].sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind="mergesort")]
        else:
            indices = np.arange(len(self))

    @property
    def num_workers(self):
        return self.datasets[0].num_workers
