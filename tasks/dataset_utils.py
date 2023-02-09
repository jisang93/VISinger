# Based on https://github.com/NATSpeech/NATSpeech
import numpy as np
import torch
import torch.distributions
import torch.optim
import torch.utils.data


from utils.audio.pitch.utils import norm_interp_f0
from utils.audio.mel_processing import SpectrogramFixed, load_wav_to_torch
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDataset


class BaseSpeechDataset(BaseDataset):
    """ Base dataset. """
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        self.data_dir = hparams["preprocess"]["binary_data_dir"] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
            if prefix == "test" and len(hparams["test_ids"]) > 0:
                self.avail_idxs = hparams["test_ids"]
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == "train" and hparams["min_frames"] > 0:
                self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] >= hparams["min_frames"]]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, "avail_idxs") and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item["mel"]) == self.sizes[index], (len(item["mel"]), self.sizes[index])
        max_frames = hparams["max_frames"]
        spec = torch.Tensor(item["mel"])[:max_frames]
        max_frames = spec.shape[0] // hparams["frames_multiple"] * hparams["frames_multiple"]
        spec = spec[:max_frames]
        ph_token = torch.LongTensor(item["ph_token"][:hparams["max_input_tokens"]])
        sample = {"id": index,
                  "item_name": item["item_name"],
                  "text": item["text"],
                  "text_token": ph_token,
                  "mel": spec,
                  "mel_nonpadding": spec.abs().sum(-1) > 0}
        if hparams["use_spk_embed"]:
            sample["spk_embed"] = torch.Tensor(item["spk_embed"])
        if hparams["use_spk_id"]:
            sample["spk_id"] = int(item["spk_id"])

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s["id"] for s in samples])
        item_names = [s["item_name"] for s in samples]
        text =[s["text"] for s in samples]
        text_tokens = collate_1d_or_2d([s["text_token"] for s in samples], 0)
        mels = collate_1d_or_2d([s["mel"] for s in samples], 0.0)
        text_lengths = torch.LongTensor([s["text_token"].numel() for s in samples])  # Return the number of total elements
        mel_lengths = torch.LongTensor([s["mel"].shape[0] for s in samples])

        batch = {"id": id,
                 "item_name": item_names,
                 "nsamples": len(samples),
                 "text": text,
                 "text_tokens": text_tokens,
                 "text_lengths": text_lengths,
                 "mels": mels,
                 "mel_lenghts": mel_lengths}

        if hparams["use_spk_embed"]:
            spk_embed = torch.stack([s["spk_embed"] for s in samples])
            batch["spk_embed"] = spk_embed
        if hparams["use_spk_id"]:
            spk_ids = torch.LongTensor([s["spk_id"] for s in samples])
            batch["spk_ids"] = spk_ids

        return batch


class VISingerDataset(BaseDataset):
    """ Dataset of VISinger. """
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        self.data_dir = hparams["binary_data_dir"] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.segment_size = hparams["segment_size"]
        self.spec_fn = SpectrogramFixed(n_fft=hparams["fft_size"], win_length=hparams["win_size"],
                                        hop_length=hparams["hop_size"], window_fn=torch.hann_window)
        self.indexed_ds = None
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
            if prefix == "test" and len(hparams["test_ids"]) > 0:
                self.avail_idxs = hparams["test_ids"]
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == "train" and self.segment_size > 0:
                self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] > self.segment_size
                                   and self.sizes[x] <= hparams["max_frames"]]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, "avail_idxs") and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        # Phoneme & Note settings
        ph_token = torch.LongTensor(item["ph_token"][:hparams["max_input_tokens"]])
        note_pitch = torch.LongTensor(item["note_pitch"][:hparams["max_input_tokens"]])
        note_dur = torch.LongTensor(item["note_duration"][:hparams["max_input_tokens"]])
        # Waveform and linear-spectrogram with max frames
        max_frames = hparams["max_frames"]
        wav, _ = load_wav_to_torch(item["wav_fn"], hparams["hop_size"])
        spec = torch.Tensor(self.spec_fn(wav)).transpose(0, 1)
        assert spec.shape[0] == self.sizes[index], (spec.shape, self.sizes[index])
        # Mapping function
        mel2ph = torch.LongTensor(item["mel2ph"][:max_frames])
        # Sample settings
        sample = {"id": index,
                  "item_name": item["item_name"],
                  "wav_fn": item["wav_fn"],
                  "text_token": ph_token,
                  "wav": wav,
                  "mel": spec,
                  "note_pitch": note_pitch,
                  "note_dur": note_dur,
                  "mel2ph": mel2ph}
        # Multi-speaker settings
        if hparams["use_spk_embed"]:
            sample["spk_embed"] = torch.Tensor(item["spk_embed"])
        if hparams["use_spk_id"]:
            sample["spk_id"] = int(item["spk_id"])
        # Pitch settings
        uv, f0 = None, None
        if hparams["use_pitch_embed"]:
            T = spec.shape[0]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
        sample["f0"], sample["uv"] = f0, uv
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s["id"] for s in samples])
        item_names = [s["item_name"] for s in samples]
        wav_fns = [s["wav_fn"] for s in samples]
        text_tokens = collate_1d_or_2d([s["text_token"] for s in samples], 0)
        text_lengths = torch.LongTensor([s["text_token"].numel() for s in samples])  # Return the number of total elements
        wavs = collate_1d_or_2d([s["wav"] for s in samples], 0.0)
        mels = collate_1d_or_2d([s["mel"] for s in samples], 0.0)
        mel_lengths = torch.LongTensor([s["mel"].shape[0] for s in samples])
        note_pitches = collate_1d_or_2d([s["note_pitch"] for s in samples], 0)
        note_durations = collate_1d_or_2d([s["note_dur"] for s in samples], 0)
        mel2phs = collate_1d_or_2d([s["mel2ph"] for s in samples], 0)
        batch = {"id": id,
                 "item_name": item_names,
                 "wav_fn": wav_fns,
                 "nsamples": len(samples),
                 "text_tokens": text_tokens,
                 "text_lengths": text_lengths,
                 "wavs": wavs,
                 "mels": mels,
                 "mel_lengths": mel_lengths,
                 "note_pitch": note_pitches,
                 "note_dur": note_durations,
                 "mel2ph": mel2phs,}
        if hparams["use_spk_embed"]:
            spk_embed = torch.stack([s["spk_embed"] for s in samples])
            batch["spk_embed"] = spk_embed
        if hparams["use_spk_id"]:
            spk_ids = torch.LongTensor([s["spk_id"] for s in samples])
            batch["spk_ids"] = spk_ids
        f0, uv, pitch = None, None, None
        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            uv = collate_1d_or_2d([s['uv'] for s in samples])
        batch.update({'pitch': pitch, 'f0': f0, 'uv': uv})
        return batch
