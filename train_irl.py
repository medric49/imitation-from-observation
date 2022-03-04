import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import hydra
from pathlib import Path
from workspaces import IRLWorkspace as W


@hydra.main(config_path='irl_cfgs', config_name='config')
def main(cfg):
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
