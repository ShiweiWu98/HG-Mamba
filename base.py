import pytorch_lightning as PL
import torch
from metrics import psnr, ssim, ciede2000
import os.path as osp
from PIL import Image
import os
from typing import Dict, Any, List
import csv
import warnings
try:
    import pyiqa
except:
    None


class Base_Trainer(PL.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.save_hyperparameters('cfg')
        self.val_metrics = MetricTracker()
        self.test_metrics = MetricTracker()
        self.pred_metrics = MetricTracker()
    
    def inference(self, x):
        pred = self.forward(x)
        return pred[-1] if isinstance(pred, list) else pred

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        smoky, real_clear = batch
        pred_clear = self.inference(smoky)

        ssim_vals = self._cal_ssim(pred_clear, real_clear)
        psnr_vals = self._cal_psnr(pred_clear, real_clear)
        ciede_vals = self._cal_ciede(pred_clear, real_clear)
        self.val_metrics.update_metrics(ssim=ssim_vals, psnr=psnr_vals,ciede2000=ciede_vals)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.get_metrics()

        avg_ssim = torch.tensor(metrics.get("ssim", [0.0])).mean().item()
        avg_psnr = torch.tensor(metrics.get("psnr", [0.0])).mean().item()
        avg_ciede = torch.tensor(metrics.get("ciede2000", [0.0])).mean().item()
        log_dict = {
            'val_metrics/ssim': avg_ssim,
            'val_metrics/psnr': avg_psnr,
            'val_metrics/ciede': avg_ciede,
        }
  
        self.log_dict(log_dict, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_metrics = MetricTracker()  

    def test_step(self, batch, batch_idx):
        smoky, real_clear = batch
        pred_clear = self.inference(smoky)
        self.test_metrics.update_metrics(
            ssim=self._cal_ssim(pred_clear, real_clear),
            psnr=self._cal_psnr(pred_clear, real_clear),
            ciede=self._cal_ciede(pred_clear, real_clear),
        )

    def on_test_epoch_end(self):
        metrics = self.test_metrics.get_metrics()
        def get_avg_std(key):
            vals = torch.tensor(metrics.get(key, [0.0]))
            return vals.mean().item(), vals.std().item()

        avg_ssim, std_ssim = get_avg_std("ssim")
        avg_psnr, std_psnr = get_avg_std("psnr")
        avg_ciede, std_ciede = get_avg_std("ciede")

        log_dict = {
            'test_metrics/ssim': avg_ssim,
            'test_metrics/psnr': avg_psnr,
            'test_metrics/ciede': avg_ciede,
            'test_metrics/ssim_std': std_ssim,
            'test_metrics/psnr_std': std_psnr,
            'test_metrics/ciede_std': std_ciede,
        }
        self.log_dict(log_dict, on_epoch=True, prog_bar=True, logger=True)
        self.test_metrics = MetricTracker()
            
    def predict_step(self, batch, batch_idx):
        smoky, smoky_path = batch
        pred_clear = self.inference(smoky)
        # metric_results = self.cal_no_ref_metrics(pred_clear)
        # self.pred_metrics.update_metrics(**metric_results)
        for idx in range(smoky.shape[0]):
            img_name = osp.basename(smoky_path[idx])
            self._save_image(pred_clear[idx],img_name)
        return 
    
    # def on_predict_epoch_end(self, results: List[Any] = None) -> None:
    #     self._save_metrics_csv(self.pred_metrics.get_metrics())
    #     self.pred_metrics = MetricTracker()      
              
    def _cal_ssim(self, img_tensor_1, img_tensor_2):
        ssim_vals = []
        for i in range(img_tensor_1.shape[0]):
            ssim_val = ssim(img_tensor_1[i].unsqueeze(
                0), img_tensor_2[i].unsqueeze(0)).item()
            ssim_vals.append(ssim_val)
        return ssim_vals

    def _cal_psnr(self, img_tensor_1, img_tensor_2):
        psnr_vals = []
        for i in range(img_tensor_1.shape[0]):
            psnr_val = psnr(img_tensor_1[i].unsqueeze(
                0), img_tensor_2[i].unsqueeze(0))
            psnr_vals.append(psnr_val)
        return psnr_vals

    def _cal_ciede(self, img_tensor_1, img_tensor_2):
        ciede_vals = []
        for i in range(img_tensor_1.shape[0]):
            ciede_val = ciede2000(img_tensor_1[i], img_tensor_2[i])
            ciede_vals.append(ciede_val)
        return ciede_vals
    
    def cal_no_ref_metrics(
            self,
        img_tensor: torch.Tensor,  # Input tensor [B,C,H,W]
        metric_names: List[str] = ['brisque_matlab','niqe_matlab','piqe','pi','nrqm'],
        clamp_input: bool = True,   # Clamp input to [0,1]
        safe_mode: bool = True      # Skip failed metrics
    ) -> Dict[str, List[float]]:
        """
        Args:
            img_tensor: Model output tensor
            metric_names: Metrics to compute
            clamp_input: Clamp input to [0,1] if True
            safe_mode: Skip metrics that fail to initialize if True
        
        Returns:
            Dictionary {metric_name: score_list}
        """
        # Validate input shape and type
        assert isinstance(img_tensor, torch.Tensor), "Input must be torch.Tensor"
        assert img_tensor.dim() == 4, "Input shape must be [B,C,H,W]"
        
        if clamp_input:
            img_tensor = img_tensor.clamp(0, 1)
        else:
            if img_tensor.min() < 0 or img_tensor.max() > 1:
                warnings.warn("Input values are outside [0,1], which may cause IQA errors.")

        # Convert to float32
        img_tensor = img_tensor.to(torch.float32)
        
        # Compute metrics
        device = img_tensor.device
        results = {}
        
        for name in metric_names:
            try:
                metric = pyiqa.create_metric(name, device=device)
                scores = metric(img_tensor)
                results[name] = scores.cpu().numpy().tolist()
                    
            except Exception as e:
                if safe_mode:
                    warnings.warn(f"Metric {name} was skipped (error: {str(e)})")
                    continue
                raise
        
        return results
    
    def _save_image(self, img_tensor, name):
        narr = img_tensor.detach().mul_(255).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
        pil_img = Image.fromarray(narr)
        path = self.hparams.cfg.common.path_to_save_image
        os.makedirs(path,exist_ok=True)
        pil_img.save(osp.join(path,name))
        
    def _save_metrics_csv(self, metrics: Dict[str, list]):
        save_dir = self.hparams.cfg.common.path_to_save_image
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"metrics.csv")

        # Compute mean and standard deviation
        stats = {}
        for key, vals in metrics.items():
            vals_tensor = torch.tensor(vals)
            stats[key] = (vals_tensor.mean().item(), vals_tensor.std().item())

        # Write metrics to CSV
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Mean", "Std"])
            for metric_name, (mean, std) in stats.items():
                writer.writerow([metric_name, f"{mean:.6f}", f"{std:.6f}"])


class MetricTracker:
    def __init__(self,metrics=None):
        # Initialize storage for metrics
        if metrics is None:
            metrics = {}
        self.metrics = metrics

    def update_metrics(self, **new_metrics: Dict[str, Any]):
        """
        Update metrics with new values.
        :param new_metrics: Metric values, each can be a scalar or a list.
        """
        for key, value in new_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            # Extend if value is a list, otherwise append
            if isinstance(value, list):
                self.metrics[key].extend(value)
            else:
                self.metrics[key].append(value)

    def get_metrics(self) -> Dict[str, list]:
        """
        Return all stored metrics.
        """
        return self.metrics