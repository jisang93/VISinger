import json
import miditoolkit
import os
import torch

from tqdm import tqdm

from models.visinger import VISinger
from preprocessor.base_preprocessor import BasePreprocessor
from preprocessor.base_binarizer import BaseBinarizer
from preprocessor.text.base_text_processor import get_text_processor_cls
from tasks.dataset_utils import VISingerDataset
from tasks.utils import load_data_preprocessor
from utils.audio.align import get_note2dur
from utils.audio.io import save_wav
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams, set_hparams


class VISingerInfer:
    def __init__(self, hparams, work_dir, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.work_dir = work_dir
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.preprocessor, self.preprocess_args = load_data_preprocessor()
        self.text_processor = get_text_processor_cls("ko_sing")
        self.ph_encoder = self.preprocessor.load_dict(self.data_dir)
        # Dictionary settings
        self.pitch_dict = json.load(open(f"{self.data_dir}/pitch_map.json"))
        self.dur_dict = json.load(open(f"{self.data_dir}/dur_map.json"))
        self.spk_map = self.preprocessor.load_spk_map(self.data_dir)
        self.ds_cls = VISingerDataset
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
    
    def build_model(self):
        dict_size = len(self.ph_encoder)
        model = VISinger(dict_size, len(self.pitch_dict), len(self.dur_dict), self.hparams)
        model.eval()
        load_ckpt(model, f"{self.hparams['work_dir']}/{self.work_dir}")
        self.gen_dir = f"{self.hparams['work_dir']}/{self.work_dir}/unseen_wav"
        os.makedirs(self.gen_dir, exist_ok=True)
        return model
    
    def preprocess_input(self, inp, divided=True, pitch_control=0):
        """
        Parameters
        ----------
        inp: dict
            {'item_name': (str, optional), 'spk_name': (str, optional), 'midi_fn': str}
        """
        midi_obj = miditoolkit.midi.parser.MidiFile(inp["midi_fn"], charset="korean")
        midi_info = BasePreprocessor.MIDI_to_encoding(midi_obj, 0, self.hparams["preprocess_args"])
        midi_info = self.text_process(midi_info, self.hparams)
        ret = []
        item = self.process_second_pass(midi_info, inp.get("spk_name", "csd"), self.ph_encoder, self.spk_map, pitch_control)
        mel2phone, mel2note, _, ph_token, _, _, item["midi_info"] = get_note2dur(item["midi_info"], self.hparams["hop_size"],
                                                                                 self.hparams["sample_rate"],
                                                                                 self.hparams["binarization_args"]["min_sil_duration"])
        BaseBinarizer.process_note(item, self.pitch_dict, self.dur_dict, self.tempo_dict, self.hparams["binarization_args"])
        item["item_name"] = f"{inp['item_name']}"
        item["spk_name"] = inp["spk_name"]
        item["ph_token"] = ph_token
        item["mel2ph"] = mel2phone
        item["mel2note"] = mel2note
        ret.append(item)
        return ret
    
    def input_to_batch(self, sample):
        item_names = [sample['item_name']]
        ph_token = torch.LongTensor(sample["ph_token"])[None, :].to(self.device)
        note_pitch = torch.LongTensor(sample["note_pitch"])[None, :].to(self.device)
        note_dur = torch.LongTensor(sample["note_duration"])[None, :].to(self.device)
        mel2ph = torch.LongTensor(sample["mel2ph"])[None, :].to(self.device)
        mel2note = torch.LongTensor(sample["mel2note"])[None, :].to(self.device)
        batch = {"item_name": item_names,
                 "spk_name": [sample["spk_name"]],
                 "text_token": ph_token,
                 "note_pitch": note_pitch,
                 "note_dur": note_dur,
                 "mel2ph": mel2ph,
                 "mel2note": mel2note,}
        if hparams["use_spk_id"]:
            batch["spk_id"] = torch.LongTensor([int(sample["spk_id"])]).to(self.device)
        return batch
    
    def forward_model(self, datasets):
        gen_dir = self.gen_dir
        for sample in tqdm(datasets):
            batch = self.input_to_batch(sample)
            output = self.model(batch["text_token"], batch["note_pitch"], batch["note_dur"], mel2ph=batch["mel2ph"], mel2note=batch["mel2note"],
                                spk_embed=batch.get("spk_embed"), spk_id=batch.get("spk_id"), infer=True)
            item_name = sample['item_name']
            wav_pred = output['wav_out'][0].detach().cpu().numpy()
            input_fn = f"{gen_dir}/spk{sample['spk_name']}_{item_name}.wav"
            save_wav(wav_pred, input_fn, self.hparams["sample_rate"], norm=self.hparams["out_wav_norm"])
    
    def text_process(self, midi_info, hparmas):
        _, midi_info = self.text_processor.process(midi_info, hparmas)
        return midi_info
    
    def divide_info(self, midi_info, preprocess_args):
        # Divide midi information
        midi_infos = []
        ph = []
        bar_info = []
        ph_bar_info = []
        phrase_idx = 0
        for i, midi in enumerate(midi_info):
            phrase_num = midi[0] // preprocess_args["max_bar"]
            if phrase_num == phrase_idx:
                bar_info.append(midi)
                ph_bar_info.extend(midi[-1])
            elif phrase_num > phrase_idx:
                midi_infos.append(bar_info)
                ph.append(" ".join(ph_bar_info))
                bar_info = []
                ph_bar_info = []
                phrase_idx += 1
                bar_info.append(midi)
                ph_bar_info.extend(midi[-1])
        midi_infos.append(bar_info)
        ph.append(" ".join(ph_bar_info))
        end_time_ = 0.0
        midi_infos_ = []
        phrase_ = []
        for _, phrase in enumerate(midi_infos):
            if len(phrase) > 0:
                assert len(phrase[-1]) == 8, f"| Wrong data construction :{phrase[-1]}"
                end_time = phrase[-1][5]
                # Time settings
                max_note_dur = 0
                for _, midi in enumerate(phrase):
                    midi_ = midi
                    midi_[4] -= end_time_
                    midi_[5] -= end_time_
                    max_note_dur = (midi_[5] - midi_[4]) if (midi_[5] - midi_[4]) > max_note_dur else max_note_dur
                    phrase_.append(midi_)
                if max_note_dur <= preprocess_args["max_note_dur"]:
                    midi_infos_.append(phrase_)
                    phrase_ = []
                    end_time_ = (end_time - 0.2)
        return midi_infos, ph
    
    def process_second_pass(self, midi_info, spk_name, ph_encoder, spk_map, pitch_control=0):
        midi_ = []
        ph_token = []
        phs = []
        for i, (bar, _, pitch, duration, start_time, end_time, tempo, ph_) in enumerate(midi_info):
            if i == 0:
                # Add [BOS] token
                phs.extend(["<BOS>"])
                ph = ph_encoder.encode("<BOS>")
                midi = [bar, 0, 0, 0, 0.0, start_time, tempo, ph, ["<BOS>"]]
                midi_.append(midi)
                ph_token.extend(ph)
            ph_ = [p for p in ph_ if p != "" and p != " "]
            phs.extend(ph_)
            ph = ph_encoder.encode(" ".join(ph_))
            if pitch > 0:
                pitch = pitch + pitch_control
            midi = [bar, i + 1, pitch, duration, start_time, end_time, tempo, ph, ph_]
            midi_.append(midi)
            ph_token.extend(ph)
            if i == len(midi_info) - 1:
                # Add [EOS] token
                phs.extend(["<EOS>"])
                ph = ph_encoder.encode("<EOS>")
                midi = [bar, i + 2, 0, 0, end_time, end_time + 0.1, tempo, ph, ["<EOS>"]]
                midi_.append(midi)
                ph_token.extend(ph)
        assert len(midi_info) < len(midi_), print(f"| Original token: {len(midi_info)}. Additional token: {len(midi_)}.")
        assert len(phs) == len(ph_token), print(f"| Phonmem token: {len(ph_token)}, Phonemes: {len(phs)}")
        midi_[-1][-1] = [ph_encoder.encode("<EOS>")] if midi_[-1][-2] == ["|"] else midi_[-1][-1]
        midi_[-1][-2] = ["<EOS>"] if midi_[-1][-2] == ["|"] else midi_[-1][-2]
        phs[-1] = ["<EOS>"] if phs[-1] == ["|"] else phs[-1]
        spk_id = spk_map[spk_name]
        return {"midi_info": midi_, "ph": phs, "ph_token": ph_token, "spk_id": spk_id}

    def inference(self, inp, pitch_control=0):
        items = self.preprocess_input(inp, pitch_control=pitch_control)
        output = self.forward_model(items)
        return output


if __name__ == "__main__":
    config = set_hparams("./config/models/visinger.yaml")
    work_dir = "svs/visinger"
    pitch_control = 0  # 1 is half-note
    generator = VISingerInfer(config, work_dir, device=0)
    midi_nm = ""  # MIDI data have to get lyrics
    generator.inference({"item_name": f"{midi_nm}_{pitch_control}",
                          "midi_fn": f"./data/source/new_svs/{midi_nm}.mid",
                          "spk_name": str(0)},
                        pitch_control=pitch_control)
 