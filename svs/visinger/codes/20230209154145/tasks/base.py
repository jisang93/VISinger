# Based on https://github.com/NATSpeech/NATSpeech
import filecmp
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim
import torch.utils
import traceback

import utils

from tasks.dataset_utils import BaseSpeechDataset
from tasks.utils import parse_dataset_configs, parse_mel_losses
from tasks.vocoder_infer.base_vocoder import BaseVocoder, get_vocoder_cls
from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav
from utils.audio.pitch_extractors import extract_pitch_simple
from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.dataset_utils import BaseConcatDataset, data_loader
from utils.commons.hparams import hparams
from utils.commons.multiprocess_utils import MultiprocessManager
from utils.commons.tensor_utils import tensors_to_scalars
from utils.nn.model_utils import print_arch
from utils.nn.seq_utils import weights_nonzero_speech
from utils.plot.plot import spec_to_figure
from utils.text.text_encoder import build_token_encoder


class SpeechBaseTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = BaseSpeechDataset
        self.vocoder = None
        self.data_dir = hparams["binary_data_dir"]
        if not hparams["use_word_input"]:
            self.token_encoder = build_token_encoder(f"{self.data_dir}/phone_set.json")
        else:
            self.token_encoder = build_token_encoder(f"{self.data_dir}/word_set.json")
        self.padding_idx = self.token_encoder.pad()
        self.eos_idx = self.token_encoder.eos()
        self.seg_idx = self.token_encoder.seg()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.mel_losses = parse_mel_losses()
        self.max_tokens, self.max_setences, \
            self.max_valid_tokens, self.max_valid_sentences = parse_dataset_configs()

    ###########################
    # datasets
    ###########################
    @data_loader
    def train_dataloader(self):
        if hparams["train_sets"] != "":
            train_sets = hparams["train_sets"].split("|")
            # Check if all train_sets have the same speaker map and dictionary
            binary_data_dir = hparams["binary_data_dir"]
            file_to_cmp = ["phone_set.json"]
            if os.path.exists(f"{binary_data_dir}/word_set.json"):
                file_to_cmp.append("word_set.json")
            if hparams["use_spk_id"]:
                file_to_cmp.append("spk_map.json")
            for f in file_to_cmp:
                for ds_name in train_sets:
                    base_file = os.path.join(binary_data_dir, f)
                    ds_file = os.path.join(ds_name, f)
                    assert filecmp.cmp(base_file, ds_file), \
                        f"{f} in {ds_name} is not same with that in {binary_data_dir}"
            print("ds_name: ", ds_name)
            train_dataset = BaseConcatDataset(
                [self.dataset_cls(prefix="train", shuffle=True, data_dir=ds_name) for ds_name in train_sets])
        else:
            train_dataset = self.dataset_cls(prefix=hparams["train_set_name"], shuffle=True)

        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_setences,
                                     endless=hparams["endless_ds"])
    
    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(prefix=hparams['valid_set_name'], shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_valid_tokens, self.max_valid_sentences,
                                     batch_by_size=False)
    
    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(prefix=hparams["test_set_name"], shuffle=False)
        self.test_dl = self.build_dataloader(
            test_dataset, False, self.max_valid_tokens, self.max_valid_sentences, batch_by_size=False)
        return self.test_dl
    
    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None,
                         required_batch_size_multiple=-1, endless=False, batch_by_size=True):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.commons.dataset_utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i:i + max_sentences])

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=dataset.collater,
                                           batch_sampler=batches,
                                           num_workers=num_workers,
                                           pin_memory=False)
    
    ###########################
    # scheduler and optimizer
    ###########################
    def build_model(self):
        self.build_tts_model()
        if hparams["load_ckpt"] != "":
            load_ckpt(self.model, hparams["load_ckpt"])
        print_arch(self.model)

        return self.model
    
    def build_tts_model(self):
        raise NotImplementedError
    
    def build_scheduler(self, optimizer):
        if hparams["scheduler"] == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                          last_epoch=self.current_epoch-1
                                                          **hparams["generator_scheduler_params"])
        elif hparams["scheduler"] == "step_lr":
            return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=500, gamma=0.998)
        else:
            raise NotImplementedError

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams["lr"],
            betas=(hparams["optimizer_adam_beta1"], hparams["optimizer_adam_beta2"]),
            weight_decay=hparams["weight_decay"])

        return optimizer
    
    ###########################
    # training and validation
    ###########################
    def _training_step(self, sample, batch_idx, _):
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output["batch_size"] = sample["mels"].size()[0]
        return total_loss, loss_output

    def run_model(self, sample, infer=False):
        """
        Parameters
        ----------
        sample: a batch of data
        infer: bool
            run in infer mode
        
        Return
        ------
        if not infer:
            return losses, model_out
        if infer:
            return model_out
        """
        raise NotImplementedError

    def validation_start(self):
        self.vocoder = get_vocoder_cls(hparams["vocoder"])()
        if self.global_step == 0:
            self.vocoder.model.remove_weight_norm()

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], model_out = self.run_model(sample)
        outputs["total_loss"] = sum(outputs["losses"].values())
        outputs["nsamples"] = sample["nsamples"]
        outputs = tensors_to_scalars(outputs)
        if self.global_step % hparams["valid_infer_interval"] == 0 \
                and batch_idx < hparams["num_valid_plots"]:
            self.save_valid_result(sample, batch_idx, model_out)
        
        return outputs

    def validation_end(self, outputs):
        self.vocoder = None
        return super(SpeechBaseTask, self).validation_end(outputs)

    def save_valid_result(self, sample, batch_idx, model_out):
        raise NotImplementedError

    ###########################
    # losses
    ###########################
    def add_mel_loss(self, mel_out, target, losses, postfix=""):
        for loss_name, lambd in self.mel_losses.items():
            losses[f"mel_{loss_name}{postfix}"] = getattr(self, f"{loss_name}_loss")(mel_out, target) * lambd

    def l1_loss(self, decoder_output, target):
        # decoder_output: [Batch, T, n_mel]
        # target: [Batch, T_mel, N_mel]
        l1_loss = F.l1_loss(decoder_output, target, reduction="none")
        weights = weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def mse_loss(self, decoder_output, target):
        # decoder_output: [Batch, T_mel, n_mel]
        # target: [Batch, T_mel, N_mel]
        assert decoder_output.shape == target.shape
        mse_loss = F.mse_loss(decoder_output, target, reduction="none")
        weights = weights_nonzero_speech(target)
        mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss

    def plot_mel(self, batch_idx, spec_out, spec_gt=None, name=None, title="", f0s=None, dur_info=None):
        vmin = hparams["mel_vmin"]
        vmax = hparams["mel_vmax"]
        if len(spec_out.shape) == 3:
            spec_out = spec_out[0]
        if isinstance(spec_out, torch.Tensor):
            spec_out = spec_out.cpu().numpy()
        if spec_gt is not None:
            if len(spec_gt.shape) == 3:
                spec_gt = spec_gt[0]
            if isinstance(spec_gt, torch.Tensor):
                spec_gt = spec_gt.cpu().numpy()
            max_len = max(len(spec_gt), len(spec_out))
            if max_len - len(spec_gt) > 0:
                spec_gt = np.pad(spec_gt, [[0, max_len - len(spec_gt)], [0, 0]], mode="constant",
                                 constant_values=vmin)
            if max_len - len(spec_out) > 0:
                spec_out = np.pad(spec_out, [[0, max_len - len(spec_out)], [0, 0]], mode="constant",
                                  constant_values=vmin)
            spec_out = np.concatenate([spec_out, spec_gt], -1)
        name = f"mel_val_{batch_idx}" if name is None else name
        self.logger.add_figure(name, spec_to_figure(
            spec_out, vmin, vmax, title=title, f0s=f0s, dur_info=dur_info), self.global_step)

    ##########################
    # testing
    ##########################
    def test_start(self):
        self.saving_result_pool = MultiprocessManager(int(os.getenv('N_PROC', os.cpu_count())))
        self.saving_results_futures = []
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        self.tgt_dir = hparams["processed_data_dir"]
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/wavs', exist_ok=True)
        os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)
        if hparams.get('save_mel_npy', False):
            os.makedirs(f'{self.gen_dir}/mel_npy', exist_ok=True)

    def test_step(self, sample, batch_idx):
        """
        Parameters
        ----------
        sample
        batch_idx
        
        Return
        ------
        """
        assert sample["text_tokens"].shape[0] == 1, "only support batch_size=1 in inference"
        outputs = self.run_model(sample, infer=True)
        text = sample["text"][0]
        item_name = sample["item_name"][0]
        tokens = sample["text_tokens"][0].cpu().numpy()
        mel_gt = sample["mels"][0].cpu().numpy()
        mel_pred = outputs["mel_out"][0].cpu().numpy()
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        base_fn = f"[{self.results_id:06d}][{item_name.replace('%', '_')}][%s]"
        if text is not None:
            base_fn += text.replace(":", "$3A")[:80]
        base_fn = base_fn.replace(" ", "_")
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred, mel_pred, base_fn % "P", gen_dir, str_phs])
        if hparams["save_gt"]:
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % "G", gen_dir, str_phs])
        print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            "item_name": item_name,
            "text": text,
            "ph_tokens": self.token_encoder.decode(tokens.tolist()),
            "wav_fn_pred": base_fn % "P",
            "wav_fn_gt": base_fn % "G",
        }

    def save_result(self, wav_out, mel, base_fn, gen_dir, str_phs=None, mel2ph=None, alignment=None):
        save_wav(wav_out, f"{gen_dir}/wavs/{base_fn}.wav", hparams["audio"]["sample_rate"],
                 norm=hparams["out_wav_norm"])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams["audio"]["mel_vmin"]
        spec_vmax = hparams["audio"]["mel_vmax"]
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        try:
            f0 = extract_pitch_simple(wav_out)
            f0 = f0 / 10 * (f0 > 0)
            plt.plot(f0, c="white", linewidth=1, alpha=0.6)
            if mel2ph is not None and str_phs is not None:
                decoded_txt = str_phs.split(" ")
                dur = mel2token_to_dur(torch.LongTensor(mel2ph)[None, :], len(decoded_txt))[0].numpy()
                dur = [0] + list(np.cumsum(dur))
                for i in range(len(dur) - 1):
                    shift = (i % 20) + 1
                    plt.text(dur[i], shift, decoded_txt[i])
                    plt.hlines(shift, dur[i], dur[i + 1], colors="b" if decoded_txt[i] != "|" else "black")
                    plt.vlines(dur[i], 0, 5, colors="b" if decoded_txt[i] != "|" else "black",
                               alpha=1, linewidth=1)
            plt.tight_layout()
            plt.savefig(f"{gen_dir}/plot/{base_fn}.png", format="png")
            plt.close(fig)
            if hparams.get("save_mel_npy", False):
                np.save(f"{gen_dir}/mel_npy/{base_fn}", mel)
            if alignment is not None:
                fig, ax = plt.subplots(figsize=(12, 16))
                im = ax.imshow(alignment, aspect="auto", origin="lower",
                               interpolation="none")
                decoded_txt = str_phs.split(" ")
                ax.set_yticks(np.arange(len(decoded_txt)))
                ax.set_yticklabels(list(decoded_txt), fontsize=6)
                fig.colorbar(im, ax=ax)
                fig.savefig(f"{gen_dir}/attn_plot/{base_fn}_attn.png", format="png")
                plt.close(fig)
        except Exception:
            traceback.print_exc()
        return None
