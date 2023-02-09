# Based on https://github.com/jaywalnut310/vits
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.commons.align_ops import expand_states
from modules.rel_transformer import RelativeEncoder, SinusoidalPositionalEmbedding
from modules.commons.utils import Embedding

DEFAULT_MAX_TARGET_POSITIONS = 2000
LRELU_SLOPE = 0.1


class TextEncoder(nn.Module):
    """ Text encoder of VISinger """
    def __init__(self, ph_dict_size, note_pitch_size, note_dur_size, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, use_pos_embed=False):
        super().__init__()
        self.dropout = p_dropout
        self.use_pos_embed = use_pos_embed
        # Input settings
        self.ph_emb = Embedding(ph_dict_size, hidden_channels)
        self.pitch_emb = Embedding(note_pitch_size, hidden_channels)
        self.dur_emb = Embedding(note_dur_size, hidden_channels)
        self.embed_scale = math.sqrt(hidden_channels)
        self.padding_idx = 0
        self.linear = nn.Linear(hidden_channels * 3, hidden_channels)
        self.text_encoder = RelativeEncoder(hidden_channels, filter_channels, n_heads, n_layers,
                                            kernel_size=kernel_size, p_dropout=p_dropout)
        # Position embedding
        if self.use_pos_embed:
            self.embed_positions = SinusoidalPositionalEmbedding(hidden_channels, 0, init_size=DEFAULT_MAX_TARGET_POSITIONS)

    def forward(self, text_tokens, pitch_tokens, dur_tokens, mel2ph):
        tgt_nonpadding = (text_tokens > 0).float().unsqueeze(1)
        # Text encoder
        token_emb = self.forward_text_embedding(text_tokens, pitch_tokens, dur_tokens, tgt_nonpadding.transpose(1, 2))
        enc_out = self.text_encoder(token_emb.transpose(1, 2), tgt_nonpadding)
        enc_out = expand_states(enc_out.transpose(1, 2), mel2ph)
        return enc_out.transpose(1, 2)
    
    def forward_text_embedding(self, text_tokens, pitch_tokens, dur_tokens, nonpadding):
        # Inputs embedding settings
        text_emb = self.ph_emb(text_tokens) * self.embed_scale
        pitch_emb = self.pitch_emb(pitch_tokens) * self.embed_scale
        dur_emb = self.dur_emb(dur_tokens) * self.embed_scale
        # Concatenation and linear projection for text encoder
        token_emb = torch.cat([text_emb, pitch_emb, dur_emb], 2)
        token_emb = self.linear(token_emb) * nonpadding
        # Use position embedding
        if self.use_pos_embed:
            pos_in = token_emb[..., 0]
            positions = self.embed_positions(token_emb.shape[0], token_emb.shape[2], pos_in)
            token_emb = token_emb + positions.transpose(1, 2)
        return token_emb * nonpadding


class FramePriorNetwork(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, gin_channels, p_dropout):
        super().__init__()
        self.hidden_channels = hidden_channels
        # Frame Prior Network
        self.encoder = RelativeEncoder(hidden_channels, filter_channels, n_heads, n_layers=n_layers,
                                       kernel_size=kernel_size, gin_channels=gin_channels, p_dropout=p_dropout)
        self.proj = nn.Conv1d(self.hidden_channels, self.hidden_channels * 2, 1)
    
    def forward(self, x, x_mask, g=None):
        if g is not None:
            g = g.transpose(1, 2)
        prior_out = self.encoder(x, x_mask, g)
        prior_out = self.proj(prior_out) * x_mask
        mu_p, logs_p = torch.split(prior_out, self.hidden_channels, dim=1)
        return mu_p, logs_p


class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, nonpadding, g=None):
        x = self.pre(x) * nonpadding
        x = self.enc(x, nonpadding, g=g)
        stats = self.proj(x) * nonpadding
        mu_q, logs_q = torch.split(stats, self.out_channels, dim=1)
        z_q = (mu_q + torch.randn_like(mu_q) * torch.exp(logs_q)) * nonpadding
        return z_q, mu_q, logs_q

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class PitchEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

    def forward(self, x, nonpadding, g=None):
        x = self.pre(x) * nonpadding
        x = self.enc(x, nonpadding, g=g)
        x = self.proj(x) * nonpadding
        return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class WaveNet(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WaveNet, self).__init__()
        assert(kernel_size % 2 == 1)
        self.hidden_channels =hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                        dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:,:self.hidden_channels,:]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts
