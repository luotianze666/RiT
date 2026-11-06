import argparse
import inspect
import os
import sys
from functools import partial
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from einops import rearrange
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import FlowMatchingModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size])


@torch.inference_mode()
def main(args):
    if len(args.ckpt_paths) != len(args.labels) or min(args.batch_size, args.steps) < 1:
        raise ValueError('Use one label per checkpoint and positive batch_size/steps.')
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = [yaml.safe_load((Path(p).parent.parent / 'config.yaml').read_text()) for p in args.ckpt_paths]
    if len({cfg['data_size'] for cfg in configs}) != 1:
        raise ValueError('All checkpoints must use the same image resolution.')
    transform = transforms.Compose([
        partial(center_crop_arr, image_size=configs[0]['data_size']), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    dataset = ImageFolder(args.dataset_path, transform=transform)
    count = args.batch_size * args.steps
    if len(dataset) < count:
        raise ValueError(f'Need at least {count} images.')

    # Fixed image order, horizontal flips and standard-normal time samples.
    rng = torch.Generator(device='cpu').manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=rng)[:count].tolist()
    flips = torch.rand(count, generator=rng) < 0.5
    time_noise = torch.randn(count, generator=rng)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=indices,
                        num_workers=args.num_workers, generator=torch.Generator().manual_seed(args.seed))

    curves = []
    for path, cfg in zip(args.ckpt_paths, configs):
        kwargs = {k: v for k, v in cfg.items() if k in inspect.signature(FlowMatchingModel).parameters}
        if kwargs.get('principal_dim', 0) > 0:
            kwargs['statistics_path'] = args.stats_path
        model = FlowMatchingModel(**kwargs).eval().requires_grad_(False)
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)['model']
        state = {k.replace('_orig_mod.', '').removeprefix('module.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        del state
        model.to(device)
        n_patches = (model.data_size // model.patch_size) ** 2
        latents, targets = [[] for _ in model.network.blocks], []
        noise_rng = torch.Generator(device='cpu').manual_seed(args.seed + 1)
        times_all = torch.sigmoid(time_noise * cfg['time_std'] + cfg['time_mean'])

        def capture(index, module, inputs, output):
            hidden = output[:, -n_patches:].flatten(0, 1)
            latents[index].append(hidden.detach().to(device='cpu', dtype=torch.float32, copy=True))

        handles = [block.register_forward_hook(partial(capture, i))
                   for i, block in enumerate(model.network.blocks)]
        with tqdm(total=args.steps + len(latents), desc=str(path), dynamic_ncols=True) as bar:
            bar.set_postfix_str('forward')
            try:
                for i, (images, labels) in enumerate(loader):
                    start = i * args.batch_size
                    mask = flips[start:start + len(images)]
                    images[mask] = images[mask].flip(-1)
                    noise = torch.randn(images.shape, generator=noise_rng, dtype=torch.float32)
                    images, noise, labels = images.to(device), noise.to(device), labels.to(device)
                    times = times_all[start:start + len(images)].to(device)
                    if model.principal_dim > 0:
                        images, noise = model.affine_proj(images), model.affine_proj(noise)
                    t = times[:, None, None, None]
                    x_t = (1 - t) * images + t * noise
                    targets.append(rearrange(x_t, 'b c (h p1) (w p2) -> (b h w) (c p1 p2)',
                                             p1=model.patch_size, p2=model.patch_size).cpu())
                    model.network(x_t, times, labels)  # One FP32 forward captures every backbone layer.
                    bar.update(1)
            finally:
                for handle in handles:
                    handle.remove()
            del model, images, noise, labels, times, t, x_t
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Concatenate all batches, then fit one affine least-squares probe per layer.
            target = torch.cat(targets).double()
            targets.clear()
            total = (target - target.mean(0, keepdim=True)).square().sum()
            if total == 0:
                raise ValueError('R2 is undefined for a constant target.')
            scores = []
            for i, chunks in enumerate(latents):
                bar.set_postfix_str(f'R2 layer {i}')
                features = torch.cat(chunks).double()
                chunks.clear()
                features = torch.cat([features, torch.ones(len(features), 1, dtype=torch.float64)], dim=1)
                rcond = max(features.shape) * torch.finfo(features.dtype).eps
                coef = torch.linalg.lstsq(features, target, rcond=rcond, driver='gelsd').solution
                residual = (target - features @ coef).square().sum()
                scores.append((1 - residual / total).item())
                del features, coef, residual
                bar.update(1)
        curves.append(scores)
        del latents, targets, target

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family': 'serif', 'pdf.fonttype': 42})
    plt.figure(figsize=(8, 5.5), dpi=150)
    for scores, label in zip(curves, args.labels):
        plt.plot(range(len(scores)), scores, label=label, marker='o', markersize=5, linewidth=1.8)
    plt.xlabel(r'Layer Index $i$', fontsize=14)
    plt.ylabel(r'Linear Reconstruction Fidelity ($R^2$ Score)', fontsize=14)
    plt.xticks(range(max(map(len, curves))))
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(4, len(curves)),
               frameon=True, fontsize=9, columnspacing=1.0)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../imagenet')
    parser.add_argument('--stats_path', default='./stats.pt')
    parser.add_argument('--ckpt_paths', nargs='+', default=[
        './RiT_ckpts/PCA768-v/ckpts/250000.pt', './RiT_ckpts/PCA128-v/ckpts/250000.pt',
        './RiT_ckpts/PCA768-x/ckpts/250000.pt', './RiT_ckpts/PCA128-x/ckpts/250000.pt'
    ])
    parser.add_argument('--labels', nargs='+', default=[
        r'$v$-prediction (Full)', r'$v$-prediction (PCA-128)',
        r'$x$-prediction (Full)', r'$x$-prediction (PCA-128)'
    ])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', default='./figures/r2_scores.pdf')
    main(parser.parse_args())