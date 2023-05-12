import torch
import torchaudio

from torchaudio.transforms import MelSpectrogram, Spectrogram


def load_wav_to_torch(full_path, hop_size=0, slice_train=False):
    wav, sampling_rate = torchaudio.load(full_path, normalize=True)
    if not slice_train:
        p = (wav.shape[-1] // hop_size + 1) * hop_size - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, p), mode="constant").data
    return wav.squeeze(0), sampling_rate


class SpectrogramFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log10 scale."""

    def __init__(self, **kwargs):
        super(SpectrogramFixed, self).__init__()
        self.torchaudio_backend = Spectrogram(**kwargs)

    def forward(self, x):
        outputs = self.torchaudio_backend(x)

        return outputs[..., :-1]


class MelSpectrogramFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log10 scale."""

    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)

    def forward(self, x):
        outputs = torch.log(self.torchaudio_backend(x) + 0.001)

        return outputs[..., :-1]


def torch_wav2spec(wav_fn, fft_size, hop_size, win_length, num_mels, fmin, fmax, sample_rate):
    """ Waveform to linear-spectrogram and mel-sepctrogram. """
    # Read wavform
    wav, sr = load_wav_to_torch(wav_fn, hop_size, slice_train=False)
    if sr != sample_rate:
        raise ValueError(f"{sr} SR doesn't match target {sample_rate} SR")
    if torch.min(wav) < -1.:
        print('min value is ', torch.min(wav))
    if torch.max(wav) > 1.:
        print('max value is ', torch.max(wav))
    # Spectrogram process
    spec_fn = SpectrogramFixed(n_fft=fft_size, win_length=win_length, hop_length=hop_size,
                               window_fn=torch.hann_window).to(device=wav.device)
    spec = spec_fn(wav)
    # Mel-spectrogram
    mel_fn = MelSpectrogramFixed(sample_rate=sample_rate, n_fft=fft_size, win_length=win_length,
                                 hop_length=hop_size, f_min=fmin, f_max=fmax, n_mels=num_mels,
                                 window_fn=torch.hann_window).to(device=wav.device)
    mel = mel_fn(wav)
    # Wav-processing
    wav = wav.squeeze(0)[:mel.shape[-1]*hop_size]
    # Check wav and spectorgram
    assert wav.shape[-1] == mel.shape[-1] * hop_size, f"| wav: {wav.shape}, spec: {spec.shape}, mel: {mel.shape}"
    assert mel.shape[-1] == spec.shape[-1], f"| wav: {wav.shape}, spec: {spec.shape}, mel: {mel.shape}"
    return {"wav": wav.cpu().detach().numpy(), "linear": spec.squeeze(0).T.cpu().detach().numpy(),
            "mel": mel.squeeze(0).T.cpu().detach().numpy()}
