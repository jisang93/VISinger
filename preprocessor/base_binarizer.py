# Based on https://github.com/NATSpeech/NATSpeech
import utils.commons.single_thread_env  # NOQA
import json
import numpy as np
import os
import random
import traceback

from functools import partial
from resemblyzer import VoiceEncoder
from tqdm import tqdm

from utils.audio.align import get_mel2note
from utils.audio.mel_processing import torch_wav2spec
from utils.audio.pitch.utils import f0_to_coarse
from utils.audio.pitch_extractors import extract_pitch_simple
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import remove_file, copy_file

np.seterr(divide="ignore", invalid="ignore")


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams["processed_data_dir"]
        self.processed_data_dir = processed_data_dir
        self.hparams = hparams
        self.binary_data_dir = hparams["binary_data_dir"]
        self.preprocess_args = hparams["preprocess_args"]
        self.binarization_args = hparams["binarization_args"]
        self.items = {}
        self.item_names = []
        if self.binarization_args["with_spk_f0_norm"]:
            self.spk_pitch_map = {}

    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        item_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        for r in tqdm(item_list, desc="Loading meta data."):
            item_name = r["item_name"]
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.binarization_args["shuffle"]:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_index_item_names(self):
        range_ = self._convert_range(self.binarization_args["train_range"])
        return self.item_names[range_[0]:range_[1]]

    @property
    def valid_index_item_names(self):
        range_ = self._convert_range(self.binarization_args["valid_range"])
        return self.item_names[range_[0]:range_[1]]

    @property
    def test_index_item_names(self) -> list:
        range_ = self._convert_range(self.binarization_args["test_range"])
        return self.item_names[range_[0]:range_[1]]

    def _convert_range(self, range_: list):
        if range_[1] == -1:
            range_[1] = len(self.item_names)
        return range_

    @property
    def train_title_item_names(self):
        return [item_name for item_name in self.item_names \
                if any(title in item_name for title in self.binarization_args["train_range"])]

    @property
    def valid_title_item_names(self):
        return [item_name for item_name in self.item_names \
                if any(title in item_name for title in self.binarization_args["valid_range"])]

    @property
    def test_title_item_names(self):
        return [item_name for item_name in self.item_names \
                if any(title in item_name for title in self.binarization_args["test_range"])]

    def meta_data(self, prefix: str, dataset_range):
        """
        Parameter
        ---------
        prefix: str
            Choose one of ["train", "valid", "test"]
        """
        if prefix == "valid":
            if dataset_range == "index":
                item_names = self.valid_index_item_names
            elif dataset_range == "title":
                item_names = self.valid_title_item_names
        elif prefix == "test":
            if dataset_range == "index":
                item_names = self.test_index_item_names
            elif dataset_range == "title":
                    item_names = self.test_title_item_names
        else:
            if dataset_range == "index":
                item_names = self.train_index_item_names
            elif dataset_range == "title":
                item_names = self.train_title_item_names
        for item_name in item_names:
            yield self.items[item_name]

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams["binary_data_dir"], exist_ok=True)
        for fn in ["phone_set.json", "spk_map.json"]:
            remove_file(f"{hparams['binary_data_dir']}/{fn}")
            copy_file(f"{hparams['processed_data_dir']}/{fn}", f"{hparams['binary_data_dir']}/{fn}")
        self.note_pitch_map = self.build_pitch_map()
        self.note_dur_map = self.build_dur_map()
        self.note_tempo_map = self.build_tempo_map()
        self.process_data("valid")
        self.process_data("test")
        self.process_data("train")

    def process_data(self, prefix: str):
        """
        Parameter
        ---------
        prefix: str
            Choose one of ["train", "valid", "test"]
        """
        data_dir = hparams["binary_data_dir"]
        meta_data = list(self.meta_data(prefix, self.binarization_args["dataset_range"]))
        process_item = partial(self.process_item, preprocess_args=self.preprocess_args,
                               binarization_args=self.binarization_args)
        builder = IndexedDatasetBuilder(f"{data_dir}/{prefix}")
        ph_lengths = []
        mel_lengths = []
        total_sec = 0
        max_sec = 0
        total_file = 0
        items = []
        args = [{"item": item, "note_pitch_map": self.note_pitch_map, "note_dur_map": self.note_dur_map,
                 "note_tempo_map": self.note_tempo_map} for item in meta_data[:len(meta_data)]]
        # Get information from audio and transcript
        for item_id, item in multiprocess_run_tqdm(process_item, args, desc="Processing data"):
            if item is not None:
                items.append(item)
                if self.binarization_args["with_spk_f0_norm"] and prefix == "train":
                    self.calculate_spk_stats(item["f0"], item["spk_id"])
        # Use pre-trained speaker embeddings
        if self.binarization_args["with_spk_embed"]:
            args = [{"wav": item["wav"]} for item in items]
            for item_id, spk_embed in multiprocess_run_tqdm(
                    self.get_spk_embed, args,
                    init_ctx_func=lambda wid: {"voice_encoder": VoiceEncoder().cuda()}, num_workers=4,
                    desc="Extracting spk embed"):
                items[item_id]["spk_embed"] = spk_embed
        
        for item in items:
            if not self.binarization_args["with_wav"] and "wav" in item:
                del item["wav"]
            mel_lengths.append(item["len"])
            assert item["len"] > 0, (item["item_name"], item["text"], item["mel2ph"])
            if "ph_len" in item:
                ph_lengths.append(item["ph_len"])
            if max_sec < item["sec"]:
                max_sec = item["sec"]
            total_sec += item["sec"]
            if "midi_info" in item:
                del item["midi_info"]
                del item["sec"]
                del item["others"]
            if not self.binarization_args["with_mel"] and "mel" in item:
                del item["mel"]
            builder.add_item(item)
        total_file += len(items)
        builder.finalize()

        if os.path.exists(f"{data_dir}/{prefix}_lengths.npy"):
            mel_lengths_ = np.load(f"{data_dir}/{prefix}_lengths.npy").tolist()
            mel_lengths_.extend(mel_lengths)
            mel_lengths = mel_lengths_
        np.save(f"{data_dir}/{prefix}_lengths.npy", mel_lengths)
        if len(ph_lengths) > 0:
            if os.path.exists(f"{data_dir}/{prefix}_ph_lenghts.npy"):
                ph_lengths_ = np.load(f"{data_dir}/{prefix}_ph_lenghts.npy").tolist()
                ph_lengths.extend(ph_lengths_)
            np.save(f"{data_dir}/{prefix}_ph_lenghts.npy", ph_lengths)
        if self.binarization_args["with_spk_f0_norm"] and prefix == "train":
            self.build_spk_pitch_map()
        print(f"| {prefix} total files: {total_file}, total duration: {total_sec:.3f}s, max duration: {max_sec:.3f}s")

    @classmethod
    def process_item(cls, item: dict, note_pitch_map, note_dur_map, note_tempo_map, preprocess_args, binarization_args: dict):
        item["ph_len"] = len(item["ph_token"])
        item_name = item["item_name"]
        wav_fn = item["wav_fn"]
        # Get Waveform and Mel-spectrogram information
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        if mel.shape[0] > 2:
            try:
                n_bos_frames, n_eos_frames = 0, 0
                if preprocess_args["use_text"] and preprocess_args["use_midi"]:
                    if binarization_args["with_midi_align"]:
                        # Align text information
                        cls.process_midi_align(item)
                # Mapping pitch and dur map of note
                cls.process_note(item, note_pitch_map, note_dur_map, note_tempo_map, preprocess_args, binarization_args)
                if binarization_args["trim_eos_bos"]:
                    n_bos_frames = item["duration"][0] if preprocess_args["use_midi"] else 0
                    n_eos_frames = item["duration"][-1] if preprocess_args["use_midi"] else mel.shape[0]
                    T = len(mel)
                    item["mel"] = mel[n_bos_frames:T - n_eos_frames]
                    item["mel2ph"] = item["mel2ph"][n_bos_frames:T - n_eos_frames]
                    item["duration"] = item["duration"][1:-1]
                    item["duration_midi"] = item["duration_midi"][1:-1]
                    item["len"] = item["mel"].shape[0]
                    item["wav"] = wav[n_bos_frames * hparams["hop_size"]:len(wav) - n_eos_frames * hparams["hop_size"]]
                if binarization_args["with_f0"]:
                    # Get pitch information
                    cls.process_pitch(item, n_bos_frames, n_eos_frames)
            except BinarizationError as e:
                print(f"| Skip item ({e}). item_name: {item_name}, wav_fm: {wav_fn}")
                return None
            except Exception as e:
                traceback.print_exc()
                print(f"| Skip item. item_name: {item_name}, wav_fm: {wav_fn}")
                return None
        return item

    @classmethod
    def process_audio(cls, wav_fn: str, res: dict, binarization_args: dict):
        # Get Mel-spectrogram
        wav2spec_dict = torch_wav2spec(wav_fn,
                                       fft_size=hparams["fft_size"],
                                       hop_size=hparams["hop_size"],
                                       win_length=hparams["win_size"],
                                       num_mels=hparams["num_mel_bins"],
                                       fmin=hparams["fmin"],
                                       fmax=hparams["fmax"],
                                       sample_rate=hparams["sample_rate"])
        mel = wav2spec_dict["mel"]
        wav = wav2spec_dict["wav"].astype(np.float16)

        #  Check Linear-spectrogram
        if binarization_args["with_linear"]:
            res["linear"] = wav2spec_dict["linear"]
        if "wav_norm" in wav2spec_dict:
            res["wav_norm"] = wav2spec_dict["wav_norm"]
        res.update({"mel": mel, "wav": wav, "sec": len(wav) / hparams["sample_rate"], "len": mel.shape[0]})
        return wav, mel

    @staticmethod
    def process_midi_align(item: dict):
        mel = item["mel"]
        midi_info = item["midi_info"]
        # Get align information and duration
        mel2phone, mel2note, duration, ph_token, ph_list, _, item["midi_info"] = get_mel2note(midi_info, mel, hparams["hop_size"],
                                                                                              hparams["sample_rate"], item["silence"])
        item["ph_token"] = ph_token
        item["text"] = ph_list
        if len(ph_list) < hparams["binarization_args"]["min_text"] or ph_list is None:
            raise BinarizationError(
                f"| Less than min text sequence: {len(item['ph_token'])}")
        if np.array(mel2phone).max() - 1 >= len(ph_token):
            raise BinarizationError(
                f"Align does not match: mel2ph.max - 1 != {np.array(mel2phone).max() - 1}, len(ph_token): {len(ph_token)}")
        item["mel2ph"] = mel2phone
        item["mel2note"] = mel2note
        item["duration"] = duration
        # Get phoneme to word information
        assert len(ph_token) == len(duration), "| phoneme : {len(ph_token)}, ph_duration : {len(duration)}"

    @staticmethod
    def process_note(item, note_pitch_map, note_dur_map, note_tempo_map, preprocess_args, binarization_args):
        dur_enc = list()
        dur_dec = list()
        for i in range(binarization_args["max_durations"]):
            for _ in range(binarization_args["pos_resolution"]):
                dur_dec.append(len(dur_enc))
                for _ in range(2 ** i):
                    dur_enc.append(len(dur_dec) - 1)
        def d2e(x):
            return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]
        if preprocess_args["use_midi"]:
            item["note_duration"] = [note_dur_map[str(d2e(note[3]))] for note in item["midi_info"]]
            item["note_pitch"] = [note_pitch_map[str(note[2])] for note in item["midi_info"]]
            item["note_tempo"] = [note_tempo_map[str(note[6])] for note in item["midi_info"]]
        else:
            item["note_duration"] = [0]
            item["note_pitch"] = [0]
            item["note_tempo"] = [0]

    @staticmethod
    def process_pitch(item: dict, n_bos_frames: int, n_eos_frames: int):
        wav, mel = item["wav"], item["mel"]
        # Get f0 from waveform
        f0 = extract_pitch_simple(wav)
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        assert len(mel) == len(f0) // hparams["f0_resolution"], (len(mel), len(f0))
        # Quantize f0 values
        pitch_coarse = f0_to_coarse(f0)
        item["f0"] = f0
        item["pitch"] = pitch_coarse
        if hparams["binarization_args"]["with_f0cwt"]:
            _, cont_logf0 = get_cont_logf0(f0)
            logf0s_mean, logf0s_std = np.mean(cont_logf0), np.std(cont_logf0)
            cont_logf0_norm = (cont_logf0 - logf0s_mean) / logf0s_std
            cwt_spec, _ = get_logf0_cwt(cont_logf0_norm)
            item["cwt_spec"] = cwt_spec
            item["cwt_mean"] = logf0s_mean
            item["cwt_std"] = logf0s_std

    def build_pitch_map(self):
        """ Using 0 to 128 notes for MIDI. """
        pitch_map = {"0": 0}
        for i, x in enumerate(range(self.hparams["note_range"][0], self.hparams["note_range"][1])):
            pitch_map[str(x)] = i + 1
        json.dump(pitch_map, open(f"{self.binary_data_dir}/pitch_map.json", "w"), ensure_ascii=False)

        return pitch_map

    def build_dur_map(self):
        """ Using max duration for MIDI. """
        dur_map = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2}
        for i, x in enumerate(range(0, 128)):
            dur_map[str(x)] = i + 4
        json.dump(dur_map, open(f"{self.binary_data_dir}/dur_map.json", "w"), ensure_ascii=False)
        return dur_map
    
    def build_tempo_map(self):
        tempo_map = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2}
        tempo_range = self.binarization_args["tempo_range"]
        for i, x in enumerate(range(tempo_range[0], tempo_range[1] + 1)):
            tempo_map[str(x)] = i + 4
        json.dump(tempo_map, open(f"{self.binary_data_dir}/tempo_map.json", "w"), ensure_ascii=False)
        return tempo_map
    
    def calculate_spk_stats(self, f0, spk_id):
        f0_min = f0[np.nonzero(f0)].min()
        f0_max = f0.max()
        if str(spk_id) in self.spk_pitch_map:
            spk_pitch_stat = self.spk_pitch_map[str(spk_id)]
            if spk_pitch_stat["min"] > f0_min:
                self.spk_pitch_map[str(spk_id)]["min"] = f0_min
            if spk_pitch_stat["max"] < f0_max:
                self.spk_pitch_map[str(spk_id)]["max"] = f0_max
        else:
            spk_pitch_stat = {}
            spk_pitch_stat["max"] = f0_max
            spk_pitch_stat["min"] = f0_min
            self.spk_pitch_map[str(spk_id)] = spk_pitch_stat
    
    def build_spk_pitch_map(self):
        spk_pitch_map = {}
        stat_map_dir = f"{self.binary_data_dir}/spk_pitch_map.json"
        if os.path.exists(stat_map_dir):
            spk_pitch_map = json.load(open(stat_map_dir, "r"))
        spk_pitch_map.update(self.spk_pitch_map)
        spk_pitch_map = {key: value for key, value in sorted(spk_pitch_map.items(), key=lambda x: int(x[0]))}
        print("| Statistics of speaker's pitch is saved.")
        json.dump(spk_pitch_map, open(stat_map_dir, "w"), ensure_ascii=False)

    @staticmethod
    def get_spk_embed(wav, ctx):
        return ctx["voice_encoder"].embed_utterance(wav.astype(float))
    
    @property
    def num_workers(self):
        return int(os.getenv("N_PROC", hparams.get("N_PROC", os.cpu_count())))
