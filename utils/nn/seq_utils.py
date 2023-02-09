# Based on https://github.com/NATSpeech/NATSpeech
import torch
import torch.nn.functional as F

from collections import defaultdict


def make_positions(tensor, padding_idx):
    """ Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx=1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully balanced
    # to both work with ONNX export and XLA.
    # In particular XLA prefers ints, cumsum defaults to output longs, and
    # ONNX doesn't know how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) *mask).long() + padding_idx


def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).to(lengths.device).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


def weights_nonzero_speech(target):
    # target: [B, T, mel]
    # Assign weight 1.0 to all labels for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # Assign a unique ID to each module instance, so that incremental state
    # is not shared across module instance
    if not hasattr(module_instance, "_instance_id"):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]
    
    return f"{module_name}.{module_instance._instance_id}.{key}"


def get_incremental_state(module, incremental_state, key):
    """ Helper for getting incremental state for an nn.Module. """
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """ Helper for setting incremental state for an nn.Module. """
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def group_hidden_by_segs(h, seg_ids, max_len):
    """
    Parameters
    ----------
    h: torch.Tensor([Batch, T_len, Hidden])
    seg_ids: torch.Tensor([Batch, T_len])
    max_len: int
    
    Return
    -------
    h_ph: torch.Tensor([Batch, T_phoneme, Hidden])
    """
    B, T, H = h.shape
    h_grouby_segs = h.new_zeros([B, max_len + 1, H]).scatter_add_(1, seg_ids[:, :, None].repeat([1, 1, H]), h)
    all_ones = h.new_ones(h.shape[:2])
    contigous_groupby_segs = h.new_zeros([B, max_len + 1]).scatter_add_(1, seg_ids, all_ones).contiguous()
    h_grouby_segs = h_grouby_segs[:, 1:]
    contigous_groupby_segs = contigous_groupby_segs[:, 1:]
    h_grouby_segs = h_grouby_segs / torch.clamp(contigous_groupby_segs[:, :, None], min=1)

    return h_grouby_segs, contigous_groupby_segs
