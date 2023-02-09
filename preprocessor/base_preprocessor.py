# Based on https://github.com/NATSpeech/NATSpeech
import json
import librosa
import os
import re
import traceback

from functools import partial
from tqdm import tqdm

from preprocessor.text.base_text_processor import get_text_processor_cls
from preprocessor.wave.base_wave_processor import get_wav_processor_cls
from utils.commons.hparams import hparams
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import copy_file, move_file, remove_file
from utils.text.text_encoder import is_sil_phoneme, build_token_encoder


class BasePreprocessor:
    def __init__(self):
        self.preprocess_args = hparams["preprocess_args"]
        self.raw_sample_rate = hparams["raw_sample_rate"]
        self.sample_rate = hparams["sample_rate"]
        text_processor = self.preprocess_args["text_processor"]
        self.text_processor = get_text_processor_cls(text_processor)
        self.raw_data_dir = hparams["raw_data_dir"]
        self.processed_dir = hparams["processed_data_dir"]
        self.spk_map_fn = os.path.join(self.processed_dir, "spk_map.json")

    def meta_data(self):
        """
        Returns
        -------
        {"item_name": str, "wav_fn": str, "midi_fn": str, "midi_obj": objective, "spk_name": str, "text_loader": None or Func}
        """
        raise NotImplementedError

    def process(self):
        processed_dir = self.processed_dir
        wav_processed_tmp_dir = os.path.join(processed_dir, "processed_tmp")
        remove_file(wav_processed_tmp_dir)
        os.makedirs(wav_processed_tmp_dir, exist_ok=True)
        wav_processed_tmp_tmp_dir = os.path.join(processed_dir, "processed_tmp", "tmp")
        remove_file(wav_processed_tmp_tmp_dir)
        os.makedirs(wav_processed_tmp_tmp_dir, exist_ok=True)
        wav_processed_dir = os.path.join(processed_dir, self.wav_processed_dirname)
        remove_file(wav_processed_dir)
        os.makedirs(wav_processed_dir, exist_ok=True)

        meta_data = list(tqdm(self.meta_data(), desc="Load meta data"))
        item_names = [d["item_name"] for d in meta_data]
        assert len(item_names) == len(set(item_names)), f"Key `item_name` should be Unique, {len(item_names)} == {len(set(item_names))}"

        # Initial settings for preprocess
        phone_list = []
        spk_names = set()
        process_item = partial(self.preprocess_first_pass,
                               text_processor=self.text_processor,
                               wav_processed_dir=wav_processed_dir,
                               wav_processed_tmp_dir=wav_processed_tmp_dir,
                               preprocess_args=self.preprocess_args,
                               raw_sample_rate=self.raw_sample_rate,
                               sample_rate=self.sample_rate)
        # Preprocess first pass
        items = []
        args = [{"item_name": item_raw["item_name"], "midi_obj": item_raw["midi_obj"], "text_raw": item_raw.get("text"),
                 "wav_fn": item_raw["wav_fn"], "spk_name": item_raw["spk_name"], "text_loader": item_raw.get("text_loader"),
                 "others": item_raw.get("others", None)} for item_raw in meta_data]
        item_id = 0
        for item_, (idx, items_) in zip(meta_data, multiprocess_run_tqdm(process_item, args, desc="Preprocess")):
            if items_ is not None:
                # Consider not divided song
                if not self.preprocess_args["divided"]:
                    for item in items_:
                        if "text_loader" in item:
                            del item["text_loader"]
                        if "midi_obj" in item:
                            del item["midi_obj"]
                        item["id"] = item_id
                        item["spk_name"] = item.get("spk_name", "<SINGLE_SPK>")
                        item["others"] = item.get("others", None)
                        phone_list += [ph for ph in item["ph"].split(" ") if ph != ""]
                        spk_names.add(item["spk_name"])
                        items.append(item)
                        item_id += 1
                else:
                    item_.update(items_)
                    item = item_
                    if "text_loader" in item_:
                        del item_["text_loader"]
                    if "midi_obj" in item_:
                        del item_["midi_obj"]
                    item["id"] = idx
                    item["spk_name"] = items_.get("spk_name", "<SINGLE_SPK>")
                    item["others"] = items_.get("others", None)
                    if self.preprocess_args["use_text"]:
                        phone_list += [ph for ph in items_["ph"].split(" ") if ph != ""]
                    spk_names.add(items_["spk_name"])
                    items.append(item)
        # Add encoded tokens
        if "<BOS>" not in phone_list:  # if midi contains lyrics
            phone_list.extend(["<BOS>", "<EOS>"])
        ph_encoder = self._phone_encoder(phone_list)
        # Mapping function
        spk_map = self.build_spk_map(spk_names)
        # Preprocess second pass
        args = [{"midi_info": item["midi_info"], "ph": item["ph"], "spk_name": item["spk_name"],
                 "ph_encoder": ph_encoder, "spk_map": spk_map} for item in items]
        for idx, item_new_kv in multiprocess_run_tqdm(self.preprocess_second_pass, args, desc="Add encoded tokens"):
            items[idx].update(item_new_kv)
        # Save metadata.json
        with open(os.path.join(processed_dir, f"{self.meta_csv_filename}.json"), "w") as f:
            f.write(re.sub(r"\n\s+([\d+\]])", r"\1", json.dumps(items, ensure_ascii=False, sort_keys=False, indent=1)))
        remove_file(wav_processed_tmp_dir)

    @classmethod
    def preprocess_first_pass(cls, item_name, midi_obj, text_raw, text_processor, wav_fn, spk_name, wav_processed_dir,
                              wav_processed_tmp_dir, preprocess_args, raw_sample_rate, sample_rate, text_loader=None, others=None):
        try:
            if text_loader is not None:
                text_raw = text_loader(text_raw)
            midi_info = []
            if preprocess_args["use_midi"]:
                # Extract midi information (Bar, Pos, Pitch, Duration, start_time, end_time, Tempo, Syllable)
                midi_info, silence, text = cls.MIDI_to_encoding(midi_obj, 0, preprocess_args)
            # text-to-phoneme
            ph = None
            if preprocess_args["use_text"]:
                # Calculate every information on notes without note duration
                ph, midi_info = cls.midi_to_ph(text_processor, midi_info, hparams)
            # Process wave data
            wav_fn, wav_align_fn = cls.process_wav(item_name, wav_fn, hparams["processed_data_dir"], wav_processed_tmp_dir, preprocess_args, sample_rate)
            ext = os.path.splitext(wav_fn)[1]
            os.makedirs(wav_processed_dir, exist_ok=True)
            new_wav_fn = os.path.join(wav_processed_dir, f"{item_name}{ext}")
            move_link_func = move_file if os.path.dirname(wav_align_fn) == wav_processed_tmp_dir else copy_file
            move_link_func(wav_align_fn, new_wav_fn)
            return {"item_name": item_name, "ph": ph, "text": text, "midi_info": midi_info, "wav_fn": new_wav_fn, "wav_align_fn": wav_align_fn,
                    "others": others, "spk_name": spk_name, "silence": silence}
        except:
            traceback.print_exc()
            print(f"| Error is caught. item_name: {item_name}.")
            return None

    @staticmethod
    def MIDI_to_encoding(midi_obj, id, preprocess_args):
        """ Following https://github.com/microsoft/muzic/blob/main/musicbert/preprocess.py

        Paramters
        ---------
        midi_obj: objective
            midi ojbective about melody instruments
        id: int
            ID of midi instruments. In the singing voice synthesis, Insturment ID is typically 0.
        preprocess_args: dictionary
            preprocess arugments

        Returns
        -------
        (Bar, Pos, Pitch, Duration, start_time, end_time, Tempo, Syllable)
        """
        trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
        # ticks to beat
        def time_to_pos(t):
            return round(t * preprocess_args["pos_resolution"] / midi_obj.ticks_per_beat)
        # Refine Time-Signature
        def time_signature_reduce(numerator, denominator):
            # reduction (when denominator is too large)
            while denominator > 2 ** preprocess_args["max_ts_denominator"] and denominator % 2 == 0 and numerator % 2 == 0:
                denominator //= 2
                numerator //= 2
            # decomposition (when length of a bar exceed max_notes_per_bar)
            while numerator > preprocess_args["max_notes_per_bar"] * denominator:
                for i in range(2, numerator + 1):
                    if numerator % i == 0:
                        numerator //= i
                        break
            return f"{numerator}/{denominator}"
        # Time-Signature settings
        ts_dict = dict()
        ts_list = list()
        for i in range(0, preprocess_args["max_ts_denominator"] + 1):  # 1 ~ 64
            for j in range(1, ((2 ** i) * preprocess_args["max_notes_per_bar"]) + 1):
                ts_dict[(j, 2 ** i)] = len(ts_dict)
                ts_list.append((j, 2 ** i))
        notes = midi_obj.instruments[id].notes
        notes.sort(key=lambda x: (x.start, x.pitch))
        notes_start_pos = [time_to_pos(note.start) for note in notes]
        if len(notes_start_pos) == 0:
            return list()
        max_pos = min(max(notes_start_pos) + 1, trunc_pos)
        # (Bar, TimeSig, Pos, Tempo)
        pos_to_info = [[None for _ in range(4)] for _ in range(max_pos)]
        tsc = midi_obj.time_signature_changes
        tpc = midi_obj.tempo_changes
        # Check time signature changes
        for i in range(len(tsc)):
            for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
                if j < len(pos_to_info):
                    pos_to_info[j][1] = time_signature_reduce(tsc[i].numerator, tsc[i].denominator)
        # Check tempo changes
        for i in range(len(tpc)):
            for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
                if j < len(pos_to_info):
                    pos_to_info[j][3] = int(tpc[i].tempo)
        for j in range(len(pos_to_info)):
            if pos_to_info[j][1] is None:
                # MIDI default time signature
                pos_to_info[j][1] = time_signature_reduce(4, 4)
            if pos_to_info[j][3] is None:
                pos_to_info[j][3] = preprocess_args["DEFAULT_TEMPO"]  # MIDI default tempo (BPM)
        cnt = 0
        bar = 0
        measure_length = None
        # Check bar number
        for j in range(len(pos_to_info)):
            ts = pos_to_info[j][1].split("/")
            if cnt == 0:
                measure_length = int(ts[0]) * preprocess_args["beat_note_factor"] * preprocess_args["pos_resolution"] // int(ts[1])
            pos_to_info[j][0] = bar  # bar number
            pos_to_info[j][2] = cnt  # position number in each bar
            cnt += 1
            if cnt >= measure_length:
                assert cnt == measure_length, f'| Invalid time signature change: position = {j}'
                cnt -= measure_length
                bar += 1
        midi_info_list = []
        start_distribution = [0] * preprocess_args["pos_resolution"]
        tick_to_time = midi_obj.get_tick_to_time_mapping()
        # Standard of silience would be wet with sixtyfourth note * min_duration_note (now: eighth note)
        midi_obj_notes = midi_obj.instruments[id].notes
        midi_obj_notes.sort(key=lambda x: x.start)
        lyrics = midi_obj.lyrics if not preprocess_args["use_mfa"] else [None] * len(midi_obj_notes)
        for i, note in enumerate(midi_obj_notes):
            if time_to_pos(note.start) >= trunc_pos:
                continue
            start_distribution[time_to_pos(note.start) % preprocess_args["pos_resolution"]] += 1
            info = pos_to_info[time_to_pos(note.start)]
            # For considering changing Time-Signature
            min_sil = tick_to_time[midi_obj.ticks_per_beat // (int(info[1][-1]) // 4 * preprocess_args["pos_resolution"]) * preprocess_args["min_sil_dur"]]
            # For wrong midi value
            if i > 0 and midi_info_list[-1][5] > note.start:
                midi_info_list[-1][3] = time_to_pos(note.start) - time_to_pos(midi_obj_notes[i - 1].start)
                midi_info_list[-1][5] = tick_to_time[note.start]
            # For adding Blank Note for SVS
            if i > 0 and tick_to_time[note.start] - midi_info_list[-1][5] >= min_sil:
                if midi_info_list[-1][7] == "" or midi_info_list[-1][7] == "|":
                    midi_info_list[-1][5] = tick_to_time[note.start]
                else:
                    midi_info_list.append([info[0], time_to_pos(note.start), 0, 0, midi_info_list[-1][5], tick_to_time[note.start], int(info[3]+0.5), "|"])
            elif i > 0 and tick_to_time[note.start] - midi_info_list[-1][5] < min_sil:
                midi_info_list[-1][5] = tick_to_time[note.start]
            # (Bar, Pos, Pitch, Duration, start_time, end_time, Tempo, Syllable)
            if i > 0 and (lyrics[i] == "" or lyrics[i] == "|") and (midi_info_list[-1][7] == "" or midi_info_list[-1][7] == "|"):
                # For merging forward blank note
                midi_info_list[-1][2] = 0
                midi_info_list[-1][5] = tick_to_time[note.end]
            else:
                if lyrics[i] is not None:
                    lyric = "|" if lyrics[i].text == "" else lyrics[i].text.replace(" ", "")
                midi_info_list.append([info[0], time_to_pos(note.start), note.pitch, time_to_pos(note.end) - time_to_pos(note.start),
                                       tick_to_time[note.start], tick_to_time[note.end], int(info[3]+0.5), lyric])
        if len(midi_info_list) == 0:
            return list()
        midi_info = []
        text = ""
        if preprocess_args["use_text"]:
            for i, midi in enumerate(midi_info_list):
                # Check (i-th start time - i-1th end time)
                if i > 0 and midi[4] - midi_info[-1][5] < min_sil:
                    midi_info[-1][5] = midi[4]
                # Remove | token repeated
                if i > 0 and midi[7] == "|" and midi_info[-1][7] == "|":
                    midi_info[-1][5] = midi[5]
                    midi_info[-1][2] = 0
                else:
                    if midi[7] == "|":
                        midi[2] = 0
                    text += " " if midi[7] == "|" else midi[7]
                    midi_info.append(midi)
            midi_info_list = midi_info
        midi_info_list.sort(key=lambda x: (x[0], x[4]))
        return midi_info_list, min_sil, text
    
    @staticmethod
    def midi_to_ph(text_processor, midi_info: str, hparmas):
        ph, midi_info = text_processor.process(midi_info, hparmas)  # generally use ko_sing
        return " ".join(ph), midi_info

    @staticmethod
    def text_to_ph(text_processor, text_raw: str, preprocess_args: dict):
        text_struct, _ = text_processor.process(text_raw, preprocess_args)  # generally use ko
        ph = [p for w in text_struct for p in w[1]]
        ph_gb_word = ["_".join(w[1]) for w in text_struct]
        return " ".join(ph), " ".join(ph_gb_word)

    @staticmethod
    def process_wav(item_name, wav_fn, processed_dir, wav_processed_tmp, preprocess_args, tgt_sample_rate):
        # Get wav processor method
        processors = [get_wav_processor_cls(v) for v in preprocess_args["wav_processors"]]
        processors = [k() for k in processors if k is not None]
        if len(processors) >= 1:
            output_fn_for_align = None
            ext = os.path.splitext(wav_fn)[1]
            input_fn = f"{wav_processed_tmp}/{item_name}{ext}"
            copy_file(wav_fn, input_fn)
            for p in processors:
                outputs = p.process(wav_fn, tgt_sample_rate, wav_processed_tmp, processed_dir, item_name)
                if len(outputs) == 3:
                    input_fn, _, output_fn_for_align = outputs
                else:
                    input_fn, _ = outputs
                return input_fn, output_fn_for_align
        else:
            return wav_fn, wav_fn

    def _phone_encoder(self, ph_set: list):
        ph_set_fn = os.path.join(self.processed_dir, "phone_set.json")
        if self.preprocess_args["reset_phone_dict"] or not os.path.exists(ph_set_fn):
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, "w"), ensure_ascii=False)
            print(f"| Build phone set: {ph_set}")
        else:
            load_ph_set = json.load(open(ph_set_fn, "r"))
            load_ph_set_ = set(load_ph_set)
            ph_set_ = set(ph_set)
            disjoint = (load_ph_set_ ^ ph_set_) & ph_set_
            load_ph_set.extend(list(sorted(disjoint)))
            ph_set = load_ph_set
            json.dump(ph_set, open(ph_set_fn, "w"), ensure_ascii=False)
            print(f"| Load phone set: {ph_set}")
        return build_token_encoder(ph_set_fn)

    @classmethod
    def preprocess_second_pass(cls, midi_info, ph, spk_name, ph_encoder, spk_map):
        midi_ = []
        ph_token = []
        phs = []
        if len(midi_info) != 0:
            for i, (bar, _, pitch, duration, start_time, end_time, tempo, ph_) in enumerate(midi_info):
                if i == 0:
                    # Add <BOS> token
                    phs.extend(["<BOS>"])
                    ph = ph_encoder.encode("<BOS>")
                    midi = (bar, 0, 0, 0, 0.0, start_time, tempo, ph, ["<BOS>"])
                    midi_.append(midi)
                    ph_token.extend(ph)
                ph_ = [p for p in ph_ if p != "" and p != " "]
                phs.extend(ph_)
                ph = ph_encoder.encode(" ".join(ph_))
                midi = (bar, i + 1, pitch, duration, start_time, end_time, tempo, ph, ph_)
                midi_.append(midi)
                ph_token.extend(ph)
                if i == len(midi_info) - 1:
                    # Add <EOS> token
                    phs.extend(["<EOS>"])
                    ph = ph_encoder.encode("<EOS>")
                    midi = (bar, i + 2, 0, 0, end_time, end_time + 0.1, tempo, ph, ["<EOS>"])
                    midi_.append(midi)
                    ph_token.extend(ph)
            assert len(midi_info) + 2 == len(midi_), print(f"| Original token: {len(midi_info)}. Additional token: {len(midi_)}.")
            assert len(phs) == len(ph_token), print(f"| Phonmem token: {len(ph_token)}, Phonemes: {len(phs)}")
        spk_id = spk_map[spk_name]

        return {"midi_info": midi_, "ph": phs, "ph_token": ph_token, "spk_id": spk_id}

    def build_spk_map(self, spk_names: str):
        if self.preprocess_args["reset_spk_dict"]:
            spk_map = {x: i for i, x in enumerate(sorted(list(spk_names)))}
        else:
            spk_map = self.load_spk_map(self.processed_dir)
            spk_set = {x: i + len(spk_map) for i, x in enumerate(sorted(list(spk_names))) if x not in spk_map}
            spk_map.update(spk_set)
        assert len(spk_map) == 0 or len(spk_map) <= hparams["num_spk"], len(spk_map)
        print(f"| Number of speakers: {len(spk_map)}, spk_map: {spk_map}")
        json.dump(spk_map, open(self.spk_map_fn, "w"), ensure_ascii=False)
        return spk_map

    def load_spk_map(self, base_dir: str):
        spk_map_fn = os.path.join(base_dir, "spk_map.json")
        spk_map = json.load(open(spk_map_fn, "r"))
        return spk_map
    
    def load_dict(self, base_dir: str):
        ph_encoder = build_token_encoder(os.path.join(base_dir, "phone_set.json"))
        return ph_encoder

    @property
    def meta_csv_filename(self):
        return "metadata"

    @property
    def wav_processed_dirname(self):
        return "wav_processed"
