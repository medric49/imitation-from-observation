
from train_ct import Workspace, Path
import hydra


def train_ct(cfg, hyperparams_str=''):
    root_dir = Path.cwd()
    workspace = Workspace(cfg, hyperparams_str)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()

@hydra.main(config_path='ct_cfgs', config_name='config')
def main(cfg):

    options = {
        "lambda_rec": [0.9, 1., 1.1],
        "lambda_align": [0.9, 1., 1.1],
        "lr": [1e-4, 1e-3, 1e-2]
    }

    for lr in options['lr']:
        for l_rec in options['lambda_rec']:
            for l_align in options['lambda_align']:
                cfg['ct_model']['lr'] = lr
                cfg['ct_model']['lambda_rec'] = l_rec
                cfg['ct_model']['lambda_align'] = l_align
                cfg['exp_id'] = f'{cfg["exp_id"]}_{lr}_{l_rec}_{l_align}'
                train_ct(cfg, hyperparams_str=f"_{lr}_{l_rec}_{l_align}")


if __name__ == '__main__':
    main()