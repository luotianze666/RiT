import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


@torch.inference_mode()
def main(args):
    if len(args.labels) != len(args.working_dirs):
        raise ValueError('Provide one label per working directory.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cov = torch.load(args.stats_path, map_location='cpu', weights_only=False)['cov']
    dim = cov.shape[0]
    if not 1 <= args.top_k <= dim:
        raise ValueError(f'top_k must be between 1 and {dim}.')
    cov = cov.to(device=device, dtype=torch.float64)
    vectors = torch.linalg.eigh(cov).eigenvectors[:, -args.top_k:].flip(1).contiguous()
    del cov

    curves, max_dim = [], 0
    for root in args.working_dirs:
        folder = Path(root)
        if (folder / 'ckpts').is_dir():
            folder = folder / 'ckpts'
        paths = sorted((p for p in folder.glob('*.pt') if p.stem.isdecimal()), key=lambda p: int(p.stem))
        if not paths:
            raise FileNotFoundError(f'No numeric checkpoints found in {folder}.')
        steps, ranks, energies = [], [], []
        for path in tqdm(paths, desc=str(folder), dynamic_ncols=True):
            # Keep the checkpoint on CPU; only the two embedding weights go to CUDA.
            state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)['model']
            weights = []
            for index in (0, 1):
                suffix = f'x_embedder.proj.{index}.weight'
                keys = [key for key in state if key.endswith(suffix)]
                if len(keys) != 1:
                    raise ValueError(f'{path}: expected exactly one {suffix}.')
                weights.append(state[keys[0]].to(device=device, dtype=torch.float64))
            weight = weights[1] @ weights[0]  # Second Linear @ first Linear.
            del state, weights
            if weight.shape[1] != dim:
                raise ValueError(f'{path}: embedding input dimension does not match stats.pt.')
            singular = torch.linalg.svdvals(weight)
            singular = singular[singular > 0]
            if not singular.numel():
                raise ValueError(f'{path}: zero embedding has undefined normalized energy.')
            p = singular / singular.sum()
            rank = torch.exp(-(p * p.log()).sum()).item()
            energy = ((weight @ vectors).square().sum() / weight.square().sum()).item()
            steps.append(int(path.stem))
            ranks.append(rank)
            energies.append(energy)
            max_dim = max(max_dim, min(weight.shape))
            del weight, singular, p
        curves.append((steps, ranks, energies))
    del vectors

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family': 'serif', 'pdf.fonttype': 42})
    path = Path(args.save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or '.pdf'
    ticks = sorted({step for steps, _, _ in curves for step in steps})
    for column, name, ylabel, ylim in [
        (1, 'rank', 'Effective Rank', (0, max_dim * 1.02)),
        (2, 'alignment', rf'PCA-subspace Energy Ratio $E_{{{args.top_k}}}$', (-0.05, 1.05))
    ]:
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
        for curve, label in zip(curves, args.labels):
            ax.plot(curve[0], curve[column], label=label, marker='o', markersize=5, linewidth=1.8)
        ax.set_xlabel('Training Steps', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_ylim(*ylim)
        if len(ticks) <= 12:
            ax.set_xticks(ticks)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:g}k' if x else '0'))
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(3, len(args.labels)),
                  frameon=True, fontsize=9.2, columnspacing=1.0)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.24)
        fig.savefig(path.with_name(path.stem + '_' + name + suffix), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_path', default='./stats.pt')
    parser.add_argument('--working_dirs', nargs='+', default=[
        './RiT_ckpts/vanilla-L-v',
        './RiT_ckpts/PCA768-v/ckpts', './RiT_ckpts/PCA128-v/ckpts',
        './RiT_ckpts/PCA768-x/ckpts', './RiT_ckpts/PCA128-x/ckpts'
    ])
    parser.add_argument('--labels', nargs='+', default=[
        r'$v$-prediction (full-L)',
        r'$v$-prediction (full)', r'$v$-prediction (PCA-128)',
        r'$x$-prediction (full)', r'$x$-prediction (PCA-128)'
    ])
    parser.add_argument('--top_k', type=int, default=280)
    parser.add_argument('--save_path', default='./figures/embedding.pdf')
    main(parser.parse_args())