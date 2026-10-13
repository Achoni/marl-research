"""Short integration check; its checkpoints and metrics are NOT research results."""
import argparse
from pathlib import Path

import torch

from .artifacts import write_json
from .benchmark import run_benchmark
from .config import EnvConfig
from .missions import build_manifest
from .study import VARIANTS
from . import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='new scratch directory')
    p.add_argument('--mujoco', action='store_true', help='also check headless MuJoCo rollout')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use a new/empty smoke directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'SMOKE_ONLY.txt').write_text('Eight-step integration runs. Not trained models or paper results.\n')
    torch.set_num_threads(1)
    # The single family is intentional: a small execution check, not validation
    # of the final study distribution, convergence, transfer, or scalability.
    cfg = EnvConfig(family='parallel')
    manifest_path = args.output / 'missions.json'
    write_json(manifest_path, build_manifest(cfg, per_family=1, name='smoke_3'))
    for variant, (pooling, controller) in VARIANTS.items():
        run = args.output / variant
        code = train.main(['--run-dir', str(run), '--steps', '8', '--rollout', '4',
            '--epochs', '1', '--hidden', '16', '--batch', '12', '--pooling', pooling,
            '--controller', controller, '--family', 'parallel', '--curriculum', 'none',
            '--eval-every', '0', '--eval-episodes', '1', '--checkpoint-every', '0'])
        if code:
            raise RuntimeError(f'{variant} smoke training failed: {code}')
        saved = torch.load(run / 'latest.pt', weights_only=True, map_location='cpu')
        if saved['global_step'] != 8 or not saved['checkpoint_validated']:
            raise RuntimeError('Smoke checkpoint budget/validation mismatch')
        run_benchmark(manifest_path, variant, args.output / 'results' / f'{variant}.json',
                      checkpoint=run / 'best.pt')
    for method in ('attention_pp_mpc', 'goal_mpc', 'lattice_mpc'):
        checkpoint = args.output / 'attention_pp' / 'best.pt' if method == 'attention_pp_mpc' else None
        run_benchmark(manifest_path, method, args.output / 'results' / f'{method}.json', checkpoint)
    if args.mujoco:
        run_benchmark(manifest_path, 'attention_mpc', args.output / 'results' / 'mujoco.json',
                      args.output / 'attention_mpc' / 'best.pt', plant='mujoco')
    print(f'Smoke check passed: {args.output}. These are NOT research results.')


if __name__ == '__main__':
    main()
