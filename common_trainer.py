from base import *
from utils.config import setup
from utils import EMACallback
from loss import *
import data_utils
from torch import optim, nn
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint as PLModelCheckpoint
import argparse
import importlib

def get_model_class(cfg):
    module = importlib.import_module(cfg.model_module)
    model_class = getattr(module, cfg.model_class)
    return model_class


class Desmoke_System(Base_Trainer):
    def __init__(self,
                 num_batches=None,
                 cfg=None,
                 ):
        super().__init__(cfg)
        self.save_hyperparameters('cfg')
        self.num_batches = num_batches
        self.net_desmoke = get_model_class(cfg)(**cfg.model)
        # print(self.net_desmoke)
            
        self.criterions = Desmoke_Loss(loss_name=cfg.train.loss_name)
        self.l1 = nn.L1Loss()
    
        
        assert len(cfg.train.loss_name) == len(cfg.train.weights)
        self.weight_dict = {key:value for key, value in zip(cfg.train.loss_name,cfg.train.weights)}
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
    def forward(self, smoky):
        pred_clear = self.net_desmoke(smoky)
        return pred_clear

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.net_desmoke.parameters(),
                               lr=self.hparams.cfg.train.lr,
                               )
        
        scheduler_cfg = self.hparams.cfg.train  
        scheduler_type = getattr(scheduler_cfg, "lr_scheduler", "cosine")  
        if scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_batches * scheduler_cfg.epochs,
                eta_min=scheduler_cfg.lr_min,
                last_epoch=-1
            )
        elif scheduler_type == "constant":
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        smoky, real_clear = batch
        pred_clear = self(smoky)
        
        loss = 0.
        loss_dict =  dict()
        if isinstance(pred_clear,list):
            real_clear_128 = F.interpolate(real_clear, scale_factor=0.5, mode='bilinear')
            real_clear_64 = F.interpolate(real_clear, scale_factor=0.25, mode='bilinear')
            desmoke_losses = self.criterions(pred_clear, [real_clear_64,real_clear_128,real_clear], smoky, None)
        else:
            desmoke_losses = self.criterions(pred_clear,real_clear,smoky, None)
        for loss_name, loss_item in desmoke_losses.items():
            loss_dict.update({('loss/'+ loss_name): loss_item})
            loss = loss + self.weight_dict[loss_name] *loss_item 
        loss_dict.update({'loss/total_loss': loss})
        
        try:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        except (AttributeError, IndexError):
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
        log_dict = {'learning_rate': lr}
        log_dict.update(loss_dict)
        self.log_dict(log_dict, on_step=False, on_epoch=True,
                      prog_bar=False, logger=True, sync_dist=True)
        return loss


def get_dataloader(cfg):

    use_trans = getattr(cfg.train,'use_trans', False)
    train_loader = data_utils.get_dataloader(dataset_root=cfg.common.train_root,
                                             img_size=cfg.common.img_size,
                                             batch_size=cfg.train.batch_size,
                                             shuffle=True,
                                             num_workers=cfg.train.num_workers,
                                             pin_memory=True,
                                             prefetch_factor=2,
                                             training=True,
                                             synthesis=True,
                                             transforms='paired_transform' if use_trans else None,
                                             )
    val_loader = data_utils.get_dataloader(cfg.common.val_root,
                                            img_size=cfg.common.img_size,
                                            batch_size=cfg.train.batch_size,
                                            shuffle=False,
                                            num_workers=cfg.train.num_workers,
                                            training=False,
                                            synthesis=True,
                                            )
    return train_loader, val_loader


def train(cfg):
    cfg.common.path_to_save_image = osp.join(
        cfg.common.path_to_save_image, cfg.train.run_name)
    if cfg.train.use_wandb:
        logger = WandbLogger(project=cfg.train.project,
                             offline=cfg.train.offline,
                             name=cfg.train.run_name,
                             resume="allow" if not cfg.train.resume_ckpt else None,
                             id=cfg.train.resume_wandb_id)
        # logger = SwanLabLogger(
        #     project=cfg.train.project,
        #     experiment_name=cfg.train.run_name,
        # )
    else:
        logger = TensorBoardLogger(
            "./tb_logs", name=cfg.train.project, version=cfg.train.run_name)
 
    train_loader, val_loader = get_dataloader(cfg)

    model = Desmoke_System(num_batches=len(train_loader),
                           cfg=cfg,
                           )

    checkpoint_callback = PLModelCheckpoint(
        dirpath=osp.join(cfg.common.output_dir,cfg.train.run_name),
        filename="desmoke_{epoch:02d}",
        every_n_epochs=cfg.train.check_val_every_n_epoch,
        save_top_k=3,
        save_last=True,
        monitor="val_metrics/ssim", #TODO:
        mode="max",
    )
    callbacks = [checkpoint_callback]
    if cfg.train.ema_decay > 0:
        ema_callback = EMACallback(decay=cfg.train.ema_decay,use_ema_weights=True)
        callbacks.append(ema_callback)
    try:
        trainer = PL.Trainer(max_epochs=cfg.train.epochs, 
                            devices=[cfg.common.device] if isinstance(cfg.common.device, int) else cfg.common.device,
                            check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
                            logger=logger,
                            precision="16-mixed",
                            callbacks=callbacks,)
    except:
        trainer = PL.Trainer(max_epochs=cfg.train.epochs, 
                    gpus=[cfg.common.device] if isinstance(cfg.common.device, int) else cfg.common.device,
                    check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
                    logger=logger,
                    callbacks=callbacks,)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=[val_loader],
                ckpt_path=cfg.train.resume_ckpt if cfg.train.resume else None)
    trainer.test(model=model, dataloaders=[val_loader],ckpt_path=checkpoint_callback.best_model_path)

def test(cfg):
    loader = data_utils.get_dataloader(cfg.common.test_root,
                                            img_size=cfg.common.img_size,
                                            batch_size=cfg.train.batch_size,
                                            shuffle=False,
                                            num_workers=cfg.train.num_workers,
                                            training=False,
                                            synthesis=True,
                                            )
    model = Desmoke_System.load_from_checkpoint(checkpoint_path=cfg.common.desmoke_ckpt,map_location='cpu',cfg=cfg)

    try:
        trainer = PL.Trainer(devices=[cfg.common.device] if isinstance(cfg.common.device, int) else cfg.common.device,)
    except:
        trainer = PL.Trainer(gpus=[cfg.common.device] if isinstance(cfg.common.device, int) else cfg.common.device,)

    trainer.test(model=model, dataloaders=[loader])  

def predict(cfg):
    cfg.common.path_to_save_image = osp.join(
        cfg.common.path_to_save_image, cfg.train.run_name)

    loader = data_utils.get_dataloader(r'your/predict/root',
                                            img_size=cfg.common.img_size,
                                            batch_size=cfg.train.batch_size,
                                            shuffle=False,
                                            num_workers=cfg.train.num_workers,
                                            training=False,
                                            synthesis=False,
                                            )

    model = Desmoke_System.load_from_checkpoint(checkpoint_path=cfg.common.desmoke_ckpt,map_location='cpu',cfg=cfg)
    try:
        trainer = PL.Trainer(gpus=[cfg.common.device] if isinstance(cfg.common.device, int) else cfg.common.device,)
    except:
        trainer = PL.Trainer(devices=[cfg.common.device] if isinstance(cfg.common.device, int) else cfg.common.device,)
    trainer.predict(model=model, dataloaders=[loader])      
    
    
def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Training", add_help=add_help)
    parser.add_argument("--config_file",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--mode",
                    type=str,
                    help="mode:train or test or predict")
    parser.set_defaults(
        config_file=r"config.yaml",
        mode='train',
    )
    return parser            
if __name__ == '__main__':
    args = get_args_parser(add_help=True).parse_args()
    cfg = setup(args)
    if args.mode == "train":
        train(cfg)
    elif args.mode == "test":
        test(cfg)
    elif args.mode == "predict":
        predict(cfg)
