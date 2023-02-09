# Based on https://github.com/NATSpeech/NATSpeech
import numpy as np

PITCH_EXTRACTOR = {}


def register_pitch_extractor(name):
    def register_pitch_extractor_(cls):
        PITCH_EXTRACTOR[name] = cls
        return cls
    
    return register_pitch_extractor_


def get_pitch_extractor(name):
    return PITCH_EXTRACTOR[name]


def extract_pitch_simple(wav):
    from utils.commons.hparams import hparams
    return extract_pitch(hparams["pitch_extractor"],
                         wav,
                         hparams["hop_size"],
                         hparams["sample_rate"],
                         f0_resolution=hparams.get("f0_resolution", 1),
                         f0_min=hparams["f0_min"],
                         f0_max=hparams["f0_max"])


def extract_pitch(extractor_name, wav_data: np.array, hop_size: int, sample_rate: int,
                  f0_resolution=1, f0_min=50, f0_max=1250, **kwargs):
    return get_pitch_extractor(extractor_name)(
        wav_data, hop_size, sample_rate, f0_resolution, f0_min, f0_max, **kwargs)


@register_pitch_extractor("parselmouth")
def parselmouth_pitch(wav_data: np.array, hop_size: int, sample_rate: int, f0_resolution: int,
                      f0_min: int, f0_max: int, voicing_threshold=0.6, *args, **kwargs):
    import parselmouth
    time_step = (hop_size // f0_resolution) / sample_rate * 1000 
    n_mel_frames = int(len(wav_data) // (hop_size // f0_resolution))
    f0_pm = parselmouth.Sound(wav_data, sample_rate).to_pitch_ac(
        time_step=time_step / 1000,
        voicing_threshold=voicing_threshold,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max).selected_array["frequency"]
    pad_size = (n_mel_frames - len(f0_pm) + 1) // 2
    f0 = np.pad(f0_pm, [[pad_size, n_mel_frames - len(f0_pm) - pad_size]], mode="constant")
    
    return f0


@register_pitch_extractor("pyworld")
def compute_f0(wav_data, hop_size, sample_rate, f0_resolution=1, f0_min=0.0, f0_max=8000,
               voicing_threshold=0.6, *args, **kwargs):
    import pyworld as pw
    time_step = (hop_size // f0_resolution) / sample_rate * 1000
    f0, t = pw.dio(wav_data.astype(np.double),
                   fs=sample_rate,
                   f0_ceil=f0_max,
                   frame_period=time_step)
    f0 = pw.stonemask(wav_data.astype(np.double), f0, t, sample_rate)
    f0 = f0[:len(wav_data)//(hop_size // f0_resolution)]
    f0 = np.maximum(f0, 1)
    f0 = f0.astype(np.float32)
    return f0
