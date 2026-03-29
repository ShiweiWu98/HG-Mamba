import os
from omegaconf import OmegaConf

def write_config(cfg, output_dir, name="config.yaml"):
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path

def get_cfg_from_args(args):
    cfg = OmegaConf.load(args.config_file)
    # cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args))
    return cfg

def setup(args,write=True):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    output_dir = os.path.join(cfg.common.output_dir,cfg.train.run_name) # save to 'ouput_dir/run_name'
    os.makedirs(output_dir, exist_ok=True)
    if write:
        write_config(cfg, output_dir)
    return cfg
