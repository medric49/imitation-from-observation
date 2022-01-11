import hydra
import torch
from pathlib import Path
from workspaces import CTWorkspace as W


@hydra.main(config_path='cfgs_ct', config_name='config')
def main(cfg):
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()
