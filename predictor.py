# Copyright (c) 2025-present, Royal Bank of Canada.
# Copyright (c) 2025-present, Kim et al.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

import os
from typing import Dict, Optional
import pickle
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.forecast import forecast
from datasets.loader import get_train_dataloader, get_test_dataloader
from utils.misc import prepare_inputs
from utils.misc import mkdir
from config import get_config_fingerprint, get_norm_method


class Predictor:
    def __init__(self, cfg, model, norm_module: Optional[torch.nn.Module] = None):
        self.cfg = cfg

        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module

        train_loader_cfg = cfg.clone()
        train_loader_cfg.TRAIN.SHUFFLE, train_loader_cfg.TRAIN.DROP_LAST = False, False
        self.train_loader = get_train_dataloader(train_loader_cfg)
        self.test_loader = get_test_dataloader(cfg)

        self.mse_all = []
        self.mae_all = []

        self.test_errors, self.train_errors = self._get_test_errors(), self._get_train_errors()

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        self.norm_module.requires_grad_(False).eval() if self.norm_module is not None else None
        log_dict = {}
        
        self.errors_all = {
            "test_mse": self.test_errors['mse'], 
            "test_mae": self.test_errors['mae'], 
            "train_mse": self.train_errors['mse'], 
            "train_mae": self.train_errors['mae'], 
        }

        results = self.get_results()
        self.save_results(results)

        self.errors_all["test_mse_all"] = self.test_errors['mse_all'].astype(float)
        self.errors_all["test_mae_all"] = self.test_errors['mae_all'].astype(float)
        
        self.save_to_npy(**self.errors_all)
        self.save_metrics(results)

        results_string = ", ".join([f"{metric}: {value:.8f}" for metric, value in results.items()])
        print(f"Baseline results | {results_string}")

        # log to W&B
        log_dict.update({f"Test/{metric}": value for metric, value in results.items()})

    @torch.no_grad()
    def _get_errors_from_dataloader(self, dataloader, tta=False, split='test') -> Dict[str, np.ndarray]:
        self.model.eval()
        self.norm_module.requires_grad_(False).eval() if self.norm_module is not None else None
        mse_all = []
        mae_all = []
        
        for inputs in tqdm(dataloader, desc='Calculating Errors'):
            enc_window_raw, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
            if self.norm_method == 'SAN':
                enc_window, statistics_pred = self.norm_module.normalize(enc_window_raw)
            else:  # Normalization from Non-stationary Transformer
                means = enc_window_raw.mean(1, keepdim=True).detach()
                enc_window = enc_window_raw - means
                stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5)
                enc_window /= stdev
            
            ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
            dec_zeros = torch.zeros_like(dec_window[:, -self.cfg.DATA.PRED_LEN:, :]).float()
            dec_window = torch.cat([dec_window[:, :self.cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()
            
            model_cfg = self.cfg.MODEL
            pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
            if model_cfg.output_attention:
                pred = pred[0]
            
            pred = pred[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:]
            
            if self.norm_method == 'SAN':
                pred = self.norm_module.de_normalize(pred, statistics_pred)
            else:  # De-Normalization from Non-stationary Transformer
                pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.cfg.DATA.PRED_LEN, 1))
                pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, self.cfg.DATA.PRED_LEN, 1))
            
            mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
            mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
            
            mse_all.append(mse)
            mae_all.append(mae)
                
        mse_all = torch.flatten(torch.concat(mse_all, dim=0)).cpu().numpy()
        mae_all = torch.flatten(torch.concat(mae_all, dim=0)).cpu().numpy()

        return {'mse': mse_all, 'mae': mae_all}

    def _get_train_errors(self):
        return self._get_errors_from_dataloader(self.train_loader, tta=False, split='train')

    def _get_test_errors(self):

        self.cur_step = self.cfg.DATA.SEQ_LEN - 2
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        is_last = False
        test_len = len(self.test_loader.dataset)

        for idx, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            
            batch_start = 0
            batch_end = 0
            
            while batch_end < len(enc_window_all):
                enc_window_first = enc_window_all[batch_start]
                
 
 
                batch_size = self.cfg.TTA.TAFAS.BATCH_SIZE
                period = batch_size - 1
                batch_end = batch_start + batch_size

                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start
                    is_last = True

                self.cur_step += batch_size
    
                inputs = enc_window_all[batch_start:batch_end], enc_window_stamp_all[batch_start:batch_end], dec_window_all[batch_start:batch_end], dec_window_stamp_all[batch_start:batch_end]
                

                pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)

                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)
                
                batch_start = batch_end
                batch_idx += 1
        assert self.cur_step == len(self.test_loader.dataset.test) - self.cfg.DATA.PRED_LEN - 1
        
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)
        
        return {
            'mse': self.mse_all.mean(),
            'mae': self.mae_all.mean(),
            'mse_all': self.mse_all,
            'mae_all': self.mae_all,
        }

    def get_results(self) -> Dict[str, float]:
        test_mse = self.test_errors['mse'].mean().astype(float)
        test_mae = self.test_errors['mae'].mean().astype(float)
        train_mse = self.train_errors['mse'].mean().astype(float)
        train_mae = self.train_errors['mae'].mean().astype(float)
        
        return {"test_mse": test_mse, 
                "test_mae": test_mae, 
                "train_mse": train_mse, 
                "train_mae": train_mae

            }

    def save_results(self, results):
        results_string = ", ".join([f"{metric}: {value:.8f}" for metric, value in results.items()])
        result_path = os.path.join(mkdir(self.cfg.RESULT_DIR), "test.txt")
        tmp_path = f"{result_path}.tmp"
        with open(tmp_path, "w") as f:
            f.write(results_string)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, result_path)

    def save_to_npy(self, **kwargs):
        for key, value in kwargs.items():
            result_path = os.path.join(self.cfg.RESULT_DIR, f"{key}.npy")
            tmp_path = f"{result_path}.tmp"
            with open(tmp_path, "wb") as f:
                np.save(f, value)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, result_path)

    def save_metrics(self, results):
        checkpoint_metadata = getattr(self.model, "_checkpoint_metadata", {})
        selection_metric = checkpoint_metadata.get("selection_metric")
        if selection_metric is None:
            selection_metric = self.cfg.MODEL.METRIC_NAMES[0]
        if selection_metric != "closed_form_fit":
            selection_metric = f"val_{selection_metric.lower()}"

        validation_metrics = checkpoint_metadata.get("validation_metrics", {})
        best_metric = checkpoint_metadata.get("best_metric")
        best_epoch = checkpoint_metadata.get("epoch")
        checkpoint_path = os.path.normpath(
            os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
        )

        metrics = {
            "schema_version": 1,
            "status": "evaluated",
            "model": self.cfg.MODEL.NAME,
            "dataset": self.cfg.DATA.NAME,
            "seq_len": self.cfg.DATA.SEQ_LEN,
            "pred_len": self.cfg.DATA.PRED_LEN,
            "seed": self.cfg.SEED,
            "data_scale": self.cfg.DATA.SCALE,
            "normalization": self.norm_method,
            "config_fingerprint": get_config_fingerprint(self.cfg),
            "selection_metric": selection_metric,
            "best_epoch": best_epoch,
            "best_val_metric": None if best_metric is None else float(best_metric),
            "best_val_mse": validation_metrics.get("MSE"),
            "best_val_mae": validation_metrics.get("MAE"),
            "checkpoint": checkpoint_path,
            "test_samples": int(len(self.mse_all)),
            "train_samples": int(len(self.train_errors["mse"])),
            "test_mse_std": float(np.std(self.mse_all)),
            "test_mse_min": float(np.min(self.mse_all)),
            "test_mse_max": float(np.max(self.mse_all)),
            "test_mae_std": float(np.std(self.mae_all)),
            "test_mae_min": float(np.min(self.mae_all)),
            "test_mae_max": float(np.max(self.mae_all)),
            **results,
        }
        result_path = os.path.join(self.cfg.RESULT_DIR, "metrics.json")
        tmp_path = f"{result_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, result_path)
