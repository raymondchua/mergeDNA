from datasets import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import hydra
from pathlib import Path
import wandb
from absl import logging
import numpy as np
import utils

# Pad is for handling different lengths. CLS is the classification token. N is for unknown base.
VOCAB = {
    "PAD": 0, "CLS": 1, "MASK": 2,
    "A": 3, "C": 4, "G": 5, "T": 6, "N": 7,
}

ID_PAD = VOCAB["PAD"]
ID_CLS = VOCAB["CLS"]

id_to_token = {v: k for k, v in VOCAB.items()}

def make_model(max_len, vocab_size, cfg):
    cfg.max_len = max_len
    cfg.vocab_size = vocab_size
    logging.info("model config: %s", cfg)
    return hydra.utils.instantiate(cfg)

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path(cfg.work_dir)
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        utils.set_seed_everywhere(cfg.seed)
        self.timer = utils.Timer()

        self.max_len = self.cfg.max_len + 1  # add one for CLS token
        self.model = make_model(self.max_len, len(VOCAB), self.cfg.models)

        # create logger
        if cfg.use_wandb:
            exp_name = "_".join(
                [
                    cfg.experiment,
                    cfg.models.name,
                    str(cfg.seed),
                ]
            )

            # get current working directory and add wandb_dir
            wandb_dir_absolute = Path.cwd()

            # convert wandb_dir_absolute to string
            wandb_dir_str = wandb_dir_absolute.as_posix()

            # log wandb_dir_str
            logging.info("wandb_dir_str: %s", wandb_dir_str)


            self._cfg_flatten = utils.dictionary_flatten(self.cfg)

            project_name = "mergeDNA"
            wandb.init(
                project=project_name,
                group=cfg.models.name,
                name=exp_name,
                config=self._cfg_flatten,
                dir=wandb_dir_str,
                mode=cfg.wandb_mode,
                settings=wandb.Settings(
                    start_method="thread"
                ),  # required for offline mode
            )

        else:
            wandb.init(mode="disabled")

        # check if data_dir exists
        data_dir_path = Path(self.cfg.data_dir)
        if not data_dir_path.exists():
            self.dataset = load_dataset("katarinagresova/Genomic_Benchmarks_human_nontata_promoters")
        else:
            self.dataset = load_dataset(self.cfg.data_dir)
        self.train_dset = self.dataset["train"]
        self.ds_train = self.train_dset.with_format("torch")


        print("OK")



    def clean_seq(self, seq: str) -> str:
        seq = seq.upper()
        return "".join(ch if ch in "ACGTN" else "N" for ch in seq)

    def encode_seq(self, seq: str, seq_len: int = 200):
        seq = self.clean_seq(seq)

        seq_len -= 1 # exclude CLS token

        # truncate/pad raw sequence to seq_len (bp length)
        if len(seq) > seq_len:
            seq = seq[:seq_len]
        else:
            seq = seq + ("N" * (seq_len - len(seq)))

        # map chars to IDs
        base_ids = [VOCAB[ch] for ch in seq]

        # add specials
        input_ids = [ID_CLS] + base_ids

        # attention mask (no PAD yet, since we fixed length already)
        attention_mask = [1] * len(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def decode_seq(self, token_ids: torch.Tensor, skip_special=True):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        if isinstance(token_ids[0], list):  # batch
            return [self.decode_seq(seq, skip_special) for seq in token_ids]

        tokens = []
        for i in token_ids:
            tok = id_to_token[i]
            if skip_special and tok in {"PAD", "CLS", "MASK"}:
                continue
            tokens.append(tok)

        return "".join(tokens)

    def eval(self, epoch: int = 0):
        dataloader_eval = DataLoader(self.ds_train, batch_size=self.cfg.batch_size, shuffle=False)
        loss, acc= [], [],
        for idx, batch in enumerate(dataloader_eval):
            input_x = []
            input_attention_mask = []
            for sequence in batch['seq']:
                seq = self.clean_seq(sequence)
                input_ids, attention_mask = self.encode_seq(seq, seq_len=self.max_len)
                input_x.append(input_ids)
                input_attention_mask.append(attention_mask)
            input_x = torch.stack(input_x).long().to(self.device)
            input_attention_mask = torch.stack(input_attention_mask).long().to(self.device)
            logs = self.model.eval(input_x, input_attention_mask, ID_PAD)
            loss.append(logs["loss"])
            acc.append(logs["acc"])
            break
        print(
            "eval | "
            f"epoch: {epoch} | "
            f"loss: {np.mean(loss):.4f} | "
            f"acc: {np.mean(acc):.4f} | "
        )

        return {
            "eval/loss": np.mean(loss),
            "eval/acc": np.mean(acc),
        }

    def train(self):
        dataloader = DataLoader(self.ds_train, batch_size=self.cfg.batch_size, shuffle=True)
        for i in range(self.cfg.training_iterations):
            elapsed_time, total_time = self.timer.reset()
            loss, acc, grad_norm, lr, num_tokens_merged = [], [], [], [], []
            loss_mtr, loss_mtr_no_local_encoder = [], []
            for idx, batch in enumerate(dataloader):
                input_x = []
                input_attention_mask = []
                for sequence in batch['seq']:
                    seq = self.clean_seq(sequence)
                    input_ids, attention_mask = self.encode_seq(seq, seq_len=self.max_len)
                    input_x.append(input_ids)
                    input_attention_mask.append(attention_mask)

                # convert input_x into pytorch long tensor
                input_x = torch.stack(input_x).long().to(self.device)
                input_attention_mask = torch.stack(input_attention_mask).long().to(self.device)
                logs = self.model.update(input_x, input_attention_mask, ID_PAD)
                loss.append(logs["total_loss"])
                loss_mtr.append(logs["loss_mtr"])
                loss_mtr_no_local_encoder.append(logs["loss_mtr_no_local_encoder"])
                acc.append(logs["acc"])
                grad_norm.append(logs["grad_norm"])
                lr.append(logs["lr"])
                num_tokens_merged.append(logs["num_tokens_merged"])
                break
            print(
                "train | " 
                f"epoch: {i} | "
                f"total_loss: {np.mean(loss):.4f} | "
                f"loss_mtr: {np.mean(loss_mtr):.4f} | "
                f"acc: {np.mean(acc):.4f} | "
                f"grad_norm: {np.mean(grad_norm):.4f} | "
                f"lr: {np.mean(lr):.2e} | "
                f"num_tokens_merged: {int(np.mean(num_tokens_merged))} | "
                f"Elapsed: {self.timer.format_time(elapsed_time)} | Total: {self.timer.format_time(total_time)}"
            )
            eval_logs = self.eval(i)
            if self.cfg.use_wandb:
                wandb_logs = {
                    "train/total_loss": np.mean(loss),
                    "train/loss_mtr": np.mean(loss_mtr),
                    "train/loss_mtr_no_local_encoder": np.mean(loss_mtr_no_local_encoder),
                    "train/acc": np.mean(acc),
                    "train/grad_norm": np.mean(grad_norm),
                    "train/lr": np.mean(lr),
                    "train/num_tokens_merged": int(np.mean(num_tokens_merged)),
                }
                wandb_logs.update(eval_logs)
                wandb.log(wandb_logs, step=i)


@hydra.main(config_path=".", config_name="train", version_base=None)
def main(cfg):
    from train import Workspace as W

    workspace = W(cfg)
    workspace.train()


if __name__ == "__main__":
    main()



