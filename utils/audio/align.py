# Based on https://github.com/NATSpeech/NATSpeech
import numpy as np
import torch


def get_mel2note(midi_info, mel, hop_size, sample_rate, min_sil_duration=0):
    # Check intervals
    midi_info_ = []
    # midi_info: (Bar, Pos, Pitch, Duration_midi, start_time, end_time, Tempo, phone_token, phone)
    for i, midi in enumerate(midi_info):
        # Check (i-th start time - i-1th end time)
        if i > 0 and midi[4] - midi_info_[-1][5] < min_sil_duration:
            midi_info_[-1][5] = midi[4]
        # Remove | token after <BOS> or before <EOS> or repeat
        if i > 0 and midi[8][0] == "|" and (midi_info_[-1][8][0] == "<BOS>" or midi_info_[-1][8][0] == "|"):
            midi_info_[-1][5] = midi[5]
            midi_info_[-1][2] = 0
        elif i > 0 and midi[8][0] == "<EOS>" and midi_info_[-1][8][0] == "|":
            midi_info_[-1][5] = midi[5]
            midi_info_[-1][2] = 0
        else:
            if midi[8][0] == "|":
                midi[2] = 0
            midi_info_.append(midi)
    # For remove, zero duration BOS token
    midi_info_ = [midi for midi in midi_info_ if not (midi[8][0] == "<BOS>" and midi[5] - midi[4] < 0.001)]
    # Check phoneme
    mel2phone = np.zeros([mel.shape[0]])
    mel2note = np.zeros([mel.shape[0]])
    ph_token_list = []
    ph_list = []
    note_token_list = []
    i_note = 0
    while i_note < len(midi_info_):
        # midi_info: (Bar, Pos, Pitch, Duration_midi, start_time, end_time, Tempo, phone_token, phone)
        midi = midi_info_[i_note]
        start_frame = int(midi[4] * sample_rate / hop_size + 0.5)
        end_frame = int(midi[5] * sample_rate / hop_size + 0.5)
        assert end_frame - start_frame > 0, f"| Wrong note: {end_frame - start_frame}"
        mel2phone[start_frame:end_frame] = i_note + 1
        mel2note[start_frame:end_frame] = i_note + 1
        ph_token_list.extend(midi[7])
        ph_list.extend(midi[8])
        note_token_list.append(midi[3])
        i_note += 1
    
    mel2phone[-1] = mel2phone[-2]
    mel2note[-1] = mel2note[-2]
    assert not np.any(mel2phone == 0) and not np.any(mel2note == 0), f"| mel2phone: {mel2phone}, mel2note: {mel2note}, midi_info: {midi_info}"
    assert mel2phone[-1] == len(ph_token_list), f"| last melphone index: {mel2phone[-1]}, length ph_list: {len(ph_token_list)}, midi_info: {len(midi_info_)}"

    T_ph = len(ph_list)
    duration = mel2token_to_dur(mel2phone, T_ph)

    return mel2phone.tolist(), mel2note.tolist(), duration.tolist(), ph_token_list, ph_list, note_token_list, midi_info_

    
def get_note2dur(midi_info, hop_size, sample_rate, min_sil_duration=0):
    # Check intervals
    midi_info_ = []
    for i, midi in enumerate(midi_info):
        # Check (i-th start time - i-1th end time)
        if i > 0 and midi[4] - midi_info_[-1][5] < min_sil_duration:
            midi_info_[-1][5] = midi[4]
        if i > 0 and midi[8] == "|" and midi_info_[-1][8] == "|":
            midi_info_[-1][5] = midi[5]
        else:
            midi_info_.append(midi)
    # Check phoneme
    last_frame = int(midi_info_[-1][5] * sample_rate / hop_size + 0.5)
    mel2phone = np.zeros([last_frame], dtype=int)
    mel2note = np.zeros([last_frame], dtype=int)
    ph_list = []
    i_note = 0
    i_ph = 0
    while i_note < len(midi_info_):
        # midi_info: (Bar, Pos, Pitch, Duration_midi, start_time, end_time, Tempo, Syllable)
        midi = midi_info_[i_note]
        start_frame = int(midi[4] * sample_rate / hop_size + 0.5)
        end_frame = int(midi[5] * sample_rate / hop_size + 0.5)
        if len(midi[7]) == 1:
            mel2phone[start_frame:end_frame] = i_ph + 1
            i_ph += 1
        elif len(midi[7]) == 2:
            mel2phone[start_frame:start_frame+3] = i_ph + 1
            mel2phone[start_frame+3:end_frame] = i_ph + 2
            i_ph += 2
        elif len(midi[7]) == 3:
            # Korean syllable consist of consonant, vowel, coda
            mel2phone[start_frame:start_frame+3] = i_ph + 1
            mel2phone[start_frame+3:end_frame-3] = i_ph + 2
            mel2phone[end_frame-3:end_frame] = i_ph + 3
            i_ph += 3
        ph_list.extend(midi[7])
        mel2note[start_frame:end_frame] = i_note + 1
        i_note += 1
    
    mel2phone[-1] = mel2phone[-2]
    mel2note[-1] = mel2note[-2]
    assert not np.any(mel2phone == 0) and not np.any(mel2note == 0), f"| mel2phone: {mel2phone}, mel2note: {mel2note}, midi_info: {midi_info}"
    T_ph = len(ph_list)
    duration = mel2token_to_dur(mel2phone, T_ph)

    return mel2phone.tolist(), mel2note.tolist(), duration.tolist(), ph_list, midi_info_


def mel2token_to_dur(mel2token: torch.Tensor, T_txt=None, max_dur=None):
    # Check input data settings
    is_torch = isinstance(mel2token, torch.Tensor)
    has_batch_dim = True
    if not is_torch:
        mel2token = torch.LongTensor(mel2token)
    if T_txt is None:
        T_txt = mel2token.max()
    if len(mel2token.shape) == 1:
        mel2token = mel2token[None, ...]
        has_batch_dim = False
    
    B, _ = mel2token.shape
    dur = mel2token.new_zeros(B, T_txt + 1).scatter_add(1, mel2token, torch.ones_like(mel2token))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    if not is_torch:
        dur = dur.numpy()
    if not has_batch_dim:
        dur = dur[0]

    return dur
