import torch
import torch.nn as nn

from copy import deepcopy

from modules.commons.utils import Embedding, rand_slice_segments
from modules.discriminator import  DiscriminatorP, DiscriminatorS
from modules.rel_transformer import SinusoidalPositionalEmbedding
from modules.visinger.encoder import TextEncoder, PosteriorEncoder, FramePriorNetwork
from modules.visinger.decoder import Generator
from modules.visinger.predictor import PitchPredictor, PhonemePredictor
from modules.visinger.flow import ResidualCouplingBlock
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse

DEFAULT_MAX_TARGET_POSITIONS = 2000


class VISinger(nn.Module):
    """ VISinger Implementation [Y. Zang et al., 2022] """
    def __init__(self, ph_dict_size, pitch_size, dur_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams["enc_layers"]
        self.dec_blocks = hparams["dec_blocks"]
        self.hidden_size = hparams["hidden_size"]
        self.use_pos_embed = hparams["use_pos_embed"]
        self.segment_size = hparams["segment_size"]
        self.out_dims = hparams["num_mel_bins"] if out_dims is None else out_dims
        # Multi-speaker settings
        if hparams["use_spk_id"]:
            self.spk_id_proj = Embedding(hparams["num_spk"], hparams["gin_channels"])
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, hparams["gin_channels"], bias=True)
        ####################
        # Prior encoder
        ####################
        # Text encoder
        self.text_encoder = TextEncoder(ph_dict_size, pitch_size, dur_size, self.hidden_size, hparams["ffn_filter_channels"],
                                        hparams["num_heads"], self.enc_layers, hparams["ffn_kernel_size"], hparams["p_dropout"], True)
        # Position encoding
        self.embed_positions = SinusoidalPositionalEmbedding(self.hidden_size, 0, init_size=DEFAULT_MAX_TARGET_POSITIONS)
        # Pitch predictor
        if hparams["use_pitch_embed"]:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            self.pitch_predictor = PitchPredictor(self.hidden_size, hparams["ffn_filter_channels"], hparams["num_heads"], 
                                                  n_layers=hparams["pitch_predictor_layers"], kernel_size=hparams['ffn_kernel_size'],
                                                  p_dropout=hparams["p_dropout"], gin_channels=hparams["gin_channels"])
        # Phoneme predictor
        if hparams["use_phoneme_pred"]:
            self.phoneme_predictor = PhonemePredictor(ph_dict_size, self.hidden_size, hparams["ffn_filter_channels"], hparams["num_heads"], 
                                                      n_layers=hparams["phoneme_predictor_layers"], kernel_size=hparams["ffn_kernel_size"],
                                                      p_dropout=hparams["p_dropout"])
        # Frame prior network
        self.frame_prior = FramePriorNetwork(self.hidden_size, hparams["ffn_filter_channels"], hparams["num_heads"], hparams["frame_prior_layers"],
                                             hparams["ffn_kernel_size"], p_dropout=hparams["p_dropout"], gin_channels=self.hidden_size)

        ####################
        # Posterior encoder
        ####################
        self.posterior_encoder = PosteriorEncoder(hparams["num_linear_bins"], self.hidden_size, self.hidden_size, 5, 1, 16,
                                                  gin_channels=hparams["gin_channels"])
        ####################
        # Generator
        ####################
        # Flow
        self.flow = ResidualCouplingBlock(self.hidden_size, self.hidden_size, 5, 1, 4, gin_channels=hparams["gin_channels"])
        # Raw Waveform Deocder
        self.decoder = Generator(self.hidden_size, hparams["dec_blocks"], hparams["dec_kernel_size"],
                                 hparams["dec_dilation_sizes"], hparams["upsample_rates"], hparams["initial_upsample_channels"],
                                 hparams["upsample_kernel_sizes"], gin_channels=hparams["gin_channels"])

    def forward(self, text_tokens, pitch_tokens, dur_tokens, mel2ph, spk_embed=None, spk_id=None, f0=None, mel=None,
                infer=False, **kwargs):
        ret = {}
        # Encoder
        tgt_nonpadding = (mel2ph > 0).float().unsqueeze(1)
        prior_inp = self.text_encoder(text_tokens, pitch_tokens, dur_tokens, mel2ph)  # [Batch, Hidden, T_len]
        prior_inp = prior_inp * tgt_nonpadding
        
        # Positional encoding
        if self.use_pos_embed:
            pos_in = prior_inp.transpose(1, 2)[..., 0]
            positions = self.embed_positions(prior_inp.shape[0], prior_inp.shape[2], pos_in)
            prior_inp = prior_inp + positions.transpose(1, 2)
        
        # Multi-speaker settings        
        spk_emb = self.speaker_embedding(spk_embed, spk_id).transpose(1, 2)
        
        # Pitch prediction
        cond_pitch = None
        if self.hparams["use_pitch_embed"]:
            cond_pitch = self.forward_pitch(prior_inp, f0, mel2ph, spk_emb, tgt_nonpadding.transpose(1, 2), ret)  # [Batch, Hidden, T_len]

        # Frame prior network
        mu_p, logs_p = self.frame_prior(prior_inp, tgt_nonpadding, cond_pitch)

        if not infer:
            # Posterior encoder
            print("mel: ", mel.shape, "tgt_nonpadding: ", tgt_nonpadding.shape)
            z_q, _, logs_q = self.posterior_encoder(mel.transpose(1, 2), tgt_nonpadding, g=spk_emb)
            # Phoneme prediction
            if self.hparams["use_phoneme_pred"]:
                ret["ph_pred"] = self.phoneme_predictor(z_q, tgt_nonpadding) * tgt_nonpadding
            # Normalizing Flow for posterior to prior
            z_p = ret["z_p"] = self.flow(z_q, tgt_nonpadding, g=spk_emb) * tgt_nonpadding # [Batch, Hidden, T_mel]
            # KL-divergence between prior from posterior and prior from prior encoder
            kl = (logs_p - logs_q - 0.5) + 0.5 * ((z_p - mu_p) ** 2) * torch.exp(-2. * logs_p)
            ret["kl"] = (kl * tgt_nonpadding).sum() / tgt_nonpadding.sum()
            # Waveform decoder
            z_slice, ret["ids_slice"] = rand_slice_segments(z_q, self.segment_size)
            ret["wav_out"] = self.decoder(z_slice, g=spk_emb).squeeze(1)
        else:
            # Reparameterization trick for prior
            z_p = (mu_p + torch.randn_like(mu_p) * torch.exp(logs_p)) * tgt_nonpadding  # [Batch, Hidden, T_len]
            # Normalizing flow for prior to posterior
            z_q = self.flow(z_p, tgt_nonpadding, g=spk_emb, reverse=True) * tgt_nonpadding
            # Waveform decoder
            ret["wav_out"] = self.decoder(z_q * tgt_nonpadding, g=spk_emb).squeeze(1)
        return ret
    
    def speaker_embedding(self, spk_embed=None, spk_id=None):
        # Add speaker embedding
        speaker_embed = 0
        if self.hparams['use_spk_embed']:
            speaker_embed = speaker_embed + self.spk_embed_proj(spk_embed)[:, None, :]
        if self.hparams['use_spk_id']:
            speaker_embed = speaker_embed + self.spk_id_proj(spk_id)[:, None, :]
        return speaker_embed
    
    def forward_pitch(self, pitch_pred_inp, f0, mel2ph, spk_emb, tgt_nonpadding, ret):
        # Pitch prediction
        pitch_padding = mel2ph == 0
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        ret['f0_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp, tgt_nonpadding.transpose(1, 2), spk_emb)
        # Teacher Forcing
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
        f0_denorm = denorm_f0(f0, None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(pitch_pred[:, :, 0], None, pitch_padding=pitch_padding)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed * tgt_nonpadding


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
