# Based on https://github.com/NATSpeech/NATSpeech
import librosa
import numpy as np
import os
import subprocess

from preprocessor.wave.base_wave_processor import BaseWavProcessor, register_wav_processors
from utils.audio import trim_long_silences
from utils.audio.io import save_wav


@register_wav_processors(name="sox_to_wav")
class ConvertToWavProcessor(BaseWavProcessor):
    @property
    def name(self):
        return "ToWav"
    
    def process(self, input_fn, sr, tmp_dir, processed_dir, item_name, preprocess_args):
        if input_fn[:-4] == ".wav":
            return input_fn,
        else:
            output_fn = self.output_fn(input_fn)
            subprocess.check_call(
                f"sox -v 0.95 '{input_fn}' -t wav '{output_fn}'",
                shell=True)
            return output_fn, sr


@register_wav_processors(name="sox_resample")
class ResampleProcessor(BaseWavProcessor):
    @property
    def name(self):
        return "Resample"
    
    def process(self, input_fn, tgt_sr, tmp_dir, processed_dir, item_name):
        output_fn = self.output_fn(input_fn)
        sr = librosa.core.get_samplerate(input_fn)
        if tgt_sr != sr:
            try:
                subprocess.check_call(f"sox '{input_fn}' -r {tgt_sr} '{output_fn}'",
                                    shell=True)
                y, _ = librosa.core.load(input_fn, sr=tgt_sr)
                save_wav(y, output_fn, tgt_sr, norm=True)
                return input_fn, sr, output_fn
            except:
                return None
        else:
            return input_fn, sr


@register_wav_processors(name="trim_sil")
class TrimSILProcessor(BaseWavProcessor):
    @property
    def name(self):
        return "TrimSIL"
    
    def process(self, input_fn, sr, tmp_dir, processed_dir, item_name, audio_args):
        output_fn = self.output_fn(input_fn)
        sr = librosa.core.get_samplerate(input_fn)
        y, _ = librosa.core.load(input_fn, sr=sr)
        y, _ = librosa.effects.tirm(y)
        save_wav(y, output_fn, sr)
        return input_fn, sr, output_fn


@register_wav_processors(name="trim_all_sil")
class TrimALLSILProcessor(BaseWavProcessor):
    @property
    def name(self):
        return "TrimALLSIL"
    
    def process(self, input_fn, sr, tmp_dir, processed_dir, item_name, preprocess_args):
        output_fn = self.output_fn(input_fn)
        y, audio_mask, _ = trim_long_silences(
                                input_fn,
                                vad_max_silence_length=preprocess_args.get("vad_max_silence_length", 12))
        save_wav(y, output_fn, sr)
        if preprocess_args["save_sil_mask"]:
            os.makedirs(f"{processed_dir}/sil_mask", exist_ok=True)
            np.save(f"{processed_dir}/sil_mask/{item_name}.npy", audio_mask)
        return output_fn, sr
