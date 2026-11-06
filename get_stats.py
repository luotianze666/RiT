import os
import argparse
from functools import partial
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from einops import rearrange
from tqdm import tqdm

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
    
def update_stats(count, mean, m2, x):
    n = x.size(0)
    batch_mean = x.mean(dim=0)
    x = x - batch_mean
    delta = batch_mean - mean
    m2.addmm_(x.T, x)
    m2.add_(torch.outer(delta, delta), alpha = count * n / (count + n))
    mean.add_(delta, alpha = n / (count + n))
    return count + n

@torch.inference_mode()
def main(args):
    assert args.patch_size > 0 and args.data_size % args.patch_size == 0
    rank, world_size, local_rank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', init_method='env://', device_id = device)

    transform = transforms.Compose([
        partial(center_crop_arr, image_size = args.data_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size = args.batch_size, sampler = range(rank, len(dataset), world_size),
                        num_workers = args.num_workers, pin_memory = True, drop_last  =False)

    p = args.patch_size
    dim = 3 * p * p
    count = 0
    mean = torch.zeros(dim, device = device, dtype = torch.float64)
    m2 = torch.zeros(dim, dim, device = device, dtype = torch.float64)
    for images, _ in tqdm(loader, disable = (rank != 0), desc = 'Patch statistics'):
        images = images.to(device = device, dtype = torch.float64, non_blocking = True)
        x = rearrange(images, 'b c (h p1) (w p2) -> (b h w) (c p1 p2)', p1 = p, p2 = p)
        count = update_stats(count, mean, m2, x)

    total = torch.tensor(count, device=device, dtype=torch.int64)
    dist.all_reduce(total)
    total_count = total.item()
    global_mean = mean * count
    dist.all_reduce(global_mean)
    global_mean.div_(total_count)
    delta = mean - global_mean
    m2.add_(torch.outer(delta, delta), alpha=count)
    dist.reduce(m2, dst=0)

    if rank == 0:
        cov = m2 / total_count
        perm = torch.arange(dim, device = device).reshape(3, p, p).flip(-1).reshape(-1)
        flipped_mean = global_mean[perm]
        delta = global_mean - flipped_mean
        mean = (global_mean + flipped_mean) * 0.5
        cov = (cov + cov[perm][:, perm]) * 0.5 + torch.outer(delta, delta) * 0.25
        cov = (cov + cov.T) * 0.5
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        torch.save({'mean': mean.cpu(), 'cov': cov.cpu(), 'num_patches': total_count,
                    'patch_size': p, 'data_size': args.data_size, 'flip_prob': 0.5}, args.save_path)
        print(f'Saved {args.save_path}: {len(dataset)} images, {total_count} patches, dim={dim}, float64, flip=0.5')
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type = str, required = True)
    parser.add_argument('--patch_size', type = int, default = 16)
    parser.add_argument('--save_path', type = str, required = True)
    parser.add_argument('--data_size', type = int, default = 256)
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--num_workers', type = int, default = 8)
    main(parser.parse_args())