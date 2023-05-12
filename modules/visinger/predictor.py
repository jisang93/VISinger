import torch.nn as nn
import torch.nn.functional as F

from modules.rel_transformer import RelativeEncoder


class PitchPredictor(nn.Module):
    """ Pitch predictor for VISinger. """
    def __init__(self, in_dim, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels, out_dim=2):
        super().__init__()
        # Pitch Predictor
        self.pitch_predictor = RelativeEncoder(in_dim, filter_channels, n_heads, n_layers=n_layers,
                                               gin_channels=gin_channels, kernel_size=kernel_size, p_dropout=p_dropout)
        self.linear = nn.Conv1d(in_dim, out_dim, 1)
    
    def forward(self, x, x_mask, spk_emb):
        x = self.pitch_predictor(x, x_mask, g=spk_emb)
        x = self.linear(x).transpose(1, 2)  # [Batch, T_len, Out_dim]
        return x


class PhonemePredictor(nn.Module):
    """ Phoneme predictor for VISinger. """
    def __init__(self, dict_size, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        # Phoneme predictor
        self.phoneme_predictor = RelativeEncoder(hidden_channels, filter_channels, n_heads, n_layers=n_layers,
                                                 kernel_size=kernel_size, p_dropout=p_dropout)
        self.ph_proj = nn.Conv1d(hidden_channels, dict_size, 1)
    
    def forward(self, x, x_mask):
        x = self.phoneme_predictor(x, x_mask)
        ph_pred = self.ph_proj(x)  # [Batch, Dict_size, T_len]
        ph_pred = F.log_softmax(ph_pred, dim=1)
        return ph_pred
