# Based on https://github.com/NATSpeech/NATSpeech
import librosa
import numpy as np
import torch

from utils.audio.mel_processing import load_wav_to_torch, MelSpectrogramFixed
from utils.commons.hparams import hparams

REGISTERED_VOCODERS = {}


def register_vocoder(name):
    def _f(cls):
        REGISTERED_VOCODERS[name] = cls
        return cls

    return _f


def get_vocoder_cls(vocoder_name):
    return REGISTERED_VOCODERS.get(vocoder_name)


class BaseVocoder:
    def spec2wav(self, mel):
        """
        Parameter
        ---------
        mel: torch.Tensor([T, 80])

        Return
        wav: torch.Tensor([T'])
        """
        raise NotImplementedError

    @staticmethod
    def wav2spec(wav_fn):
        """
        Parameter
        ---------
        wav_fn: str

        Return
        ------
        wav, mel: torch.Tensor([T, 80])
        """
        wav = load_wav_to_torch(wav_fn, hop_size=hparams['hop_size'])
        mel_fn = MelSpectrogramFixed(sample_rate=hparams["sample_rate"], n_fft=hparams["fft_size"],
                                    win_length=hparams["win_size"], hop_length=hparams["hop_size"],
                                    f_min=hparams["fmin"], f_max=hparams["fmax"], n_mels=hparams["num_mel_bins"],
                                    window_fn=torch.hann_window).to(device=wav.device)
        mel = mel_fn(wav)
        return wav, mel

    @staticmethod
    def wav2mfcc(wav_fn):
        fft_size = hparams["audio"]['fft_size']
        hop_size = hparams["audio"]['hop_size']
        win_length = hparams["audio"]['win_size']
        sample_rate = hparams["audio"]['audio_sample_rate']
        wav, _ = librosa.core.load(wav_fn, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=13,
                                    n_fft=fft_size, hop_length=hop_size,
                                    win_length=win_length, pad_mode="constant", power=1.0)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta]).T
        return mfcc
