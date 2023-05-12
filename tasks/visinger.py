import json
import os
import torch
import torch.nn.functional as F

from models.visinger import VISinger, MultiPeriodDiscriminator
from modules.commons.utils import slice_segments
from tasks.base import SpeechBaseTask
from tasks.dataset_utils import VISingerDataset
from utils.audio.io import save_wav
from utils.audio.mel_processing import MelSpectrogramFixed, SpectrogramFixed
from utils.commons.hparams import hparams
from utils.commons.multiprocess_utils import MultiprocessManager
from utils.commons.tensor_utils import tensors_to_scalars
from utils.nn.model_utils import num_params


class VISingerTask(SpeechBaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = VISingerDataset
        self.sil_ph = self.token_encoder.sil_phonemes()
        data_dir = hparams["binary_data_dir"]
        # Discriminator settings
        self.build_disc_model()
        # Dictionary settings
        self.pitch_dict = json.load(open(f"{data_dir}/pitch_map.json"))
        self.dur_dict = json.load(open(f"{data_dir}/dur_map.json"))
        # Spectrogram Function
        self.spec_fn = SpectrogramFixed(n_fft=hparams["fft_size"], win_length=hparams["win_size"],
                                        hop_length=hparams["hop_size"], window_fn=torch.hann_window)
        self.mel_fn = MelSpectrogramFixed(sample_rate=hparams["sample_rate"], n_fft=hparams["fft_size"],
                                          win_length=hparams["win_size"], hop_length=hparams["hop_size"],
                                          f_min=hparams["fmin"], f_max=hparams["fmax"],
                                          n_mels=hparams["num_mel_bins"], window_fn=torch.hann_window)

    def build_tts_model(self):
        ph_dict_size = len(self.token_encoder)
        self.model = VISinger(ph_dict_size, len(self.pitch_dict), len(self.dur_dict), hparams)
    
    def build_disc_model(self):
        self.mel_disc = MultiPeriodDiscriminator(hparams["use_spectral_norm"])
        self.disc_params = list(self.mel_disc.parameters())
    
    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        if hasattr(self.model, "visinger"):
            for n, m in self.model.visinger.named_children():
                num_params(m, model_name=f"visinger.{n}")
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["disc_start_steps"] and hparams["lambda_mel_adv"] > 0
        if optimizer_idx == 0:
            #########################
            #       Generator       #
            #########################
            loss_output, model_out = self.run_model(sample, infer=False)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                slice_wavs = slice_segments(sample["wavs"].unsqueeze(1), model_out["ids_slice"] * hparams["hop_size"],
                                            hparams["segment_size"] * hparams["hop_size"])
                _, mel_disc_gen, fmap_tgt, fmap_gen = self.mel_disc(slice_wavs, model_out["wav_out"].unsqueeze(1))
                if mel_disc_gen is not None:
                    loss_output["generator"] = self.add_generator_loss(mel_disc_gen)
                    loss_weights["generator"] = hparams["lambda_mel_adv"]
                if fmap_tgt is not None and fmap_gen is not None:
                    loss_output["feature_match"] = self.add_feature_matching_loss(fmap_tgt, fmap_gen)
                    loss_weights["feature_match"] = hparams["lambda_fm"]
        else:
            #########################
            #     Discriminator     #
            #########################
            if disc_start and self.global_step % hparams["disc_interval"] == 0:
                model_out = self.model_out_gt
                # Slicing wavs
                slice_wavs = slice_segments(sample["wavs"].unsqueeze(1), model_out["ids_slice"] * hparams["hop_size"],
                                            hparams["segment_size"] * hparams["hop_size"])
                mel_disc_tgt, mel_disc_gen, _, _ = self.mel_disc(slice_wavs, model_out["wav_out"].unsqueeze(1))
                if mel_disc_gen is not None:
                    loss_output["discriminator"] = self.add_discriminator_loss(mel_disc_tgt, mel_disc_gen)
                    loss_weights["discriminator"] = 1.0
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output["batch_size"] = sample["text_tokens"].shape[0]
        return total_loss, loss_output
    
    def run_model(self, sample, infer=False, *args):
        text_tokens = sample["text_tokens"]  # [B, T_text]
        pitch_tokens = sample["note_pitch"]
        dur_tokens = sample["note_dur"]
        mel2ph = sample["mel2ph"]  # [Batch, T_mels]
        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")
        if not infer:
            f0 = sample.get('f0')
            mel = sample["mels"]
            output = self.model(text_tokens, pitch_tokens, dur_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id, f0=f0, mel=mel, infer=False)
            # Losses
            losses = {}
            # KL divergence losses
            losses["kl_v"] = output["kl"].detach()
            losses_kl = output["kl"]
            losses_kl = torch.clamp(losses_kl, min=hparams["kl_min"])
            losses_kl = min(self.global_step / hparams["kl_start_steps"], 1) * losses_kl
            losses_kl = losses_kl * hparams["lambda_kl"]
            losses["kl"] = losses_kl
            sample["tgt_mel"] = self.mel_fn(sample["wavs"])  # [Batch, mel_bins, T_len]
            tgt_slice_mel = slice_segments(sample["tgt_mel"], output["ids_slice"], hparams["segment_size"]) # [Batch,  mel_bins, T_slice]
            output["mel_out"] = mel_out = self.mel_fn(output["wav_out"].squeeze(1))  # [Batch, mel_bins, T_slice]
            self.add_mel_loss(mel_out.transpose(1, 2), tgt_slice_mel.transpose(1, 2), losses)
            # Pitch losses
            if hparams["use_pitch_embed"]:
                self.add_pitch_loss(output, sample, losses)
            # CTC loss
            if hparams["use_phoneme_pred"]:
                self.add_ctc_loss(output, sample, losses)
            return losses, output
        else:
            output = self.model(text_tokens, pitch_tokens, dur_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                                spk_id=spk_id, f0=None, mel=None, infer=True)
            return output
    
    def add_pitch_loss(self, output, sample, losses):
        f0 = sample['f0']
        uv = sample["uv"]
        nonpadding = (sample["mel2ph"] != 0).float()  # [B, T_mels]
        p_pred = output['f0_pred']
        assert p_pred[..., 0].shape == f0.shape, f"| f0_diff: {f0.shape}, pred_diff: {p_pred.shape}"
        # Loss for voice/unvoice flag
        losses["uv"] = (F.binary_cross_entropy_with_logits(p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_uv']
        nonpadding = nonpadding * (uv == 0).float()
        # Loss for f0 difference
        losses["f0"] = (F.l1_loss(p_pred[:, :, 0], f0, reduction="none") * nonpadding).sum() \
                            / nonpadding.sum() * hparams["lambda_f0"]

    def add_ctc_loss(self, output, sample, losses):
        ph_pred = output["ph_pred"].float().permute(2, 0, 1)  # [T_mel, Batch, Dict_size]
        input_length = sample["mel_lengths"]
        text_tokens = sample["text_tokens"]  # [Batch, T_ph]
        target_length = sample["text_lengths"]
        losses["ctc"] = F.ctc_loss(ph_pred, text_tokens, input_length, target_length, zero_infinity=True) * hparams["lambda_ctc"]
    
    def add_discriminator_loss(self, tgt_output, gen_output):
        disc_loss = 0 
        for tgt, gen in zip(tgt_output, gen_output):
            r_loss = torch.mean((1 - tgt.float()) ** 2)  # (D(y) - 1)^2
            g_loss = torch.mean(gen.float() ** 2)  # (D(G(z)))^2
            disc_loss += (r_loss + g_loss)
        return disc_loss
    
    def add_generator_loss(self, gen_output):
        gen_loss = 0
        for gen in gen_output:
            gen = gen.float()
            gen_loss += torch.mean((1 - gen) ** 2)  # (D(G(z) - 1)^2
        return gen_loss

    def add_feature_matching_loss(self, fmap_tgt, fmap_gen):
        feature_loss = 0
        for tgt, gen in zip(fmap_tgt, fmap_gen):
            for tgt_layer, gen_layer in zip(tgt, gen):
                tgt_layer = tgt_layer.float().detach()
                gen_layer = gen_layer.float()
                feature_loss += torch.mean(torch.abs(tgt_layer - gen_layer))
        return feature_loss

    def validation_start(self):
        pass
    
    def save_valid_result(self, sample, batch_idx, model_out):
        sr = hparams["sample_rate"]
        mel_out = model_out["mel_out"]
        self.plot_mel(batch_idx, mel_out, sample["tgt_mel"].transpose(1, 2))
        if self.global_step > 0:
            self.logger.add_audio(f"wav_val_{batch_idx}", model_out["wav_out"], self.global_step, sr)
            if "wav_full" in model_out:
                self.logger.add_audio(f"wav_val_full_{batch_idx}", model_out["wav_full"], self.global_step, sr)
        # Ground truth wavforms
        if self.global_step <= hparams["valid_infer_interval"]:
            self.logger.add_audio(f"wav_gt_{batch_idx}", sample["wavs"], self.global_step, sr)
        
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], model_out = self.run_model(sample)
        outputs["total_loss"] = sum(outputs["losses"].values())
        outputs["nsamples"] = sample["nsamples"]
        outputs = tensors_to_scalars(outputs)
        if self.global_step % hparams["valid_infer_interval"] == 0 \
                and batch_idx < hparams["num_valid_plots"]:
            model_out = self.run_model(sample, infer=True)
            model_out["mel_out"] = self.mel_fn(model_out["wav_out"].squeeze(1)).transpose(1, 2)
            self.save_valid_result(sample, batch_idx, model_out)
        return outputs
    
    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(self.model.parameters(),
                                          lr=hparams["lr"],
                                          betas=(hparams["optimizer_adam_beta1"], hparams["optimizer_adam_beta2"]),
                                          weight_decay=hparams["weight_decay"],
                                          eps=hparams["eps"])
        optimizer_disc = torch.optim.AdamW(self.disc_params,
                                           lr=hparams["lr"],
                                           betas=(hparams["optimizer_adam_beta1"], hparams["optimizer_adam_beta2"]),
                                           **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None
        return [optimizer_gen, optimizer_disc]
    
    def build_scheduler(self, optimizer):
        return [torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer[0], # Generator Scheduler
                                                       last_epoch=self.current_epoch-1,
                                                       **hparams["generator_scheduler_params"]), 
                torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer[1], # Discriminator Scheduler
                                                       last_epoch=self.current_epoch-1,
                                                       **hparams["discriminator_scheduler_params"])]

    def on_after_optimization(self, epoch, batch_idx, optimizer, opt_idx):
        if self.scheduler is not None and hparams["endless_ds"]:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])
        elif self.scheduler is not None and not hparams["endless_ds"]:
            self.scheduler[0].step(epoch)
            self.scheduler[1].step(epoch)

    ##############################
    # inference
    ##############################
    def test_start(self):
        self.saving_result_pool = MultiprocessManager(int(os.getenv('N_PROC', os.cpu_count())))
        self.saving_results_futures = []
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        self.tgt_dir = hparams["processed_data_dir"]
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/wavs', exist_ok=True)
        os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)
        if hparams.get('save_mel_npy', False):
            os.makedirs(f'{self.gen_dir}/mel_npy', exist_ok=True)

    def test_step(self, sample, batch_idx):
        import time
        assert sample['text_tokens'].shape[0] == 1, 'only support batch_size=1 in inference'
        # Inference for test step
        start_time = time.time()
        output = self.run_model(sample, infer=True)
        running_time = time.time() - start_time
        # Inference settings
        item_name = sample['item_name'][0]
        wav_fn = f"{hparams['processed_data_dir']}/{'/'.join(sample['wav_fn'][0].split('/')[-2:])}"
        tokens = sample['text_tokens'][0].cpu().numpy()
        wav_pred = output['wav_out'][0].cpu().numpy()
        gen_dir = self.gen_dir
        input_fn = f"{gen_dir}/wavs/{item_name}_synth.wav"
        save_wav(wav_pred, input_fn, hparams["sample_rate"], norm=hparams["out_wav_norm"])
        return {'item_name': item_name,
                'ph_tokens': self.token_encoder.decode(tokens.tolist()),
                "wav_fn": wav_fn,
                'wav_fn_pred': f"{item_name}_synth.wav",
                "rtf": running_time}

    def test_end(self):
        pass
