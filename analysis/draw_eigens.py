
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(args):
    # 1. Load PCA eigenvalues
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats = torch.load(args.stats_path, map_location=device, weights_only=False)
    cov = stats['cov'].double()
    eigenvalues = torch.linalg.eigvalsh(cov).flip(0).cpu().numpy()
    eigenvalues = np.clip(eigenvalues, a_min=1e-5, a_max=None)

    # 2. Map principal component indices to equally spaced positions
    dim = len(eigenvalues)
    key_x = np.array(sorted(set([1, dim] + [2 ** i for i in range(4, dim.bit_length())])))
    visual_x = np.arange(len(key_x))
    real_indices = np.arange(1, dim + 1)
    mapped_x = np.interp(real_indices, key_x, visual_x)
    key_y = eigenvalues[key_x - 1]

    # 3. Plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(8, 6), dpi=150)

    plt.plot(mapped_x, eigenvalues, color='#1a9988', linewidth=2.75, zorder=1)
    plt.scatter(visual_x, key_y, color='red', s=55, zorder=2,
                edgecolor='white', linewidth=0.4, marker='o')

    plt.xlabel('Principal Component Index', fontsize=14)
    plt.ylabel('Eigenvalue Magnitude', fontsize=14)

    plt.xticks(visual_x, key_x)
    plt.xlim(-0.1, len(key_x) - 0.9)

    # 4. Automatic logarithmic y-axis
    plt.yscale('log')
    y_min, y_max = eigenvalues.min() * 0.6, eigenvalues.max() * 1.5
    plt.ylim(y_min, y_max)
    powers = np.arange(np.ceil(np.log10(y_min)), np.floor(np.log10(y_max)) + 1)
    plt.yticks(10.0 ** powers)

    plt.grid(True, which='major', linestyle=':', alpha=0.5)
    plt.grid(False, which='minor')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.save_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_path', default='./stats.pt')
    parser.add_argument('--save_path', default='./figures/pca_eigenvalues.pdf')
    main(parser.parse_args())
