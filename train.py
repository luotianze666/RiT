import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import matplotlib.pyplot as plt
import numpy as np
from models import FlowMatchingModel
import tqdm
from typing import Tuple, Literal, Optional
import torch.distributed as dist
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import argparse
import time
import logging
import yaml
import shutil

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
    
class Learner:
    def __init__(self,
                 working_dir: str = './your_working_dir',
                 data_channels: int = 3,
                 data_size: int = 256,
                 patch_size: int = 16,
                 hidden_size: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 num_classes: int = 1000,
                 bottleneck_dim: int = 128,
                 in_context_len: int = 32,
                 in_context_start: int = 4,
                 patch_size_final: int = 8,
                 hidden_size_final: int = 384,
                 depth_final: int = 2,
                 num_heads_final: int = 6,
                 decoder_input: bool = True, 
                 shortcut_type: str = 'fraction_anchored',
                 use_F_sdpa: bool = True,
                 use_v_pred: bool = True, 
                 principal_dim: int = 0,
                 cfg_drop_rate: float = 0.1,
                 statistics_path: Optional[str] = None,
                 dataset_path: str = './imagenet',
                 time_mean: float = 0.0, time_std: float = 1.0,
                 use_compile: bool = True, use_bf16_autocast: bool = True, 
                 total_steps: int = 250000, warmup_steps: int = 6250, ema_decay_rate: Tuple[float] = [0.9996,0.9998,0.9999],
                 learning_rate: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.999), weight_decay: float = 0.0, 
                 shortcut_lrdecay: Optional[Tuple[int, float]] = None, batch_size: int = 1024, num_workers: int = 12, 
                 prefetch_factor: int = 6, eval_steps: int = 25000, eval_batch_size: int = 250, eval_solver_steps: int = 50, 
                 eval_solver: Literal['Euler', 'Heun'] = 'Euler',
                 eval_demo_steps: int = 1000, eval_demo_shape: Tuple[int, int] = (4, 8),
                 eval_cfg_scale: float = 2.5, eval_cfg_interval: Tuple[float, float] = (0.0, 0.9),
                 logging_steps: int = 100, ckpt_saving_steps: int = 25000, seed: int = 42
                ):
        
        world_size = int(os.environ['WORLD_SIZE'])
        assert batch_size % world_size == 0, 'batch size shoule be multiple of GPU amount'
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.manual_seed(seed + rank)
        
        if rank == 0:
            os.makedirs(working_dir, exist_ok = True)
            logging.basicConfig(
                level=logging.INFO,
                format='[%(asctime)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(working_dir,'log.txt'))]
            )
            self.logger= logging.getLogger(__name__)
            self.logger.info(f"Logger initialized with world size = {world_size}")
        
        self.device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend='nccl', init_method='env://', rank = rank, world_size = world_size, device_id = self.device)
        torch.cuda.set_device(self.device)

        torch.backends.cuda.enable_flash_sdp(False)  
        torch.backends.cuda.enable_mem_efficient_sdp(True) 
        torch.backends.cuda.enable_math_sdp(True) 
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, data_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.train_set = ImageFolder(dataset_path, transform = self.transform)

        self.sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_set,
            num_replicas = world_size,
            rank = rank,
            shuffle = True
        )
            
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size = batch_size // world_size,
            sampler = self.sampler,
            num_workers = num_workers,
            prefetch_factor = prefetch_factor,
            persistent_workers=True,
            pin_memory = True,
            drop_last = True
        )
        self.working_dir = working_dir
        self.time_mean = time_mean
        self.time_std = time_std
        self.cfg_drop_rate = cfg_drop_rate
        
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.eval_solver_steps = eval_solver_steps
        self.eval_solver = eval_solver
        self.eval_cfg_scale = eval_cfg_scale
        self.eval_cfg_interval = eval_cfg_interval
        
        self.eval_time = [1.0 - i / self.eval_solver_steps for i in range(self.eval_solver_steps + 1)]
        self.eval_demo_steps = eval_demo_steps
        self.eval_demo_shape = eval_demo_shape
        
        self.learning_rate = learning_rate
        self.use_compile = use_compile
        self.use_bf16_autocast = use_bf16_autocast
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.ema_decay_rate = ema_decay_rate
        self.first_ema = self.ema_decay_rate[0]
        self.finished_steps = 0 
        self.finished_epochs = 0
        self.logging_steps = logging_steps
        self.ckpt_saving_steps = ckpt_saving_steps
        self.shortcut_lrdecay = shortcut_lrdecay
        
        self.model = FlowMatchingModel(data_channels, data_size,
                        patch_size, hidden_size, depth, num_heads,
                        mlp_ratio, attn_drop, proj_drop, num_classes,
                        bottleneck_dim, in_context_len, in_context_start,
                        patch_size_final, hidden_size_final,
                        depth_final, num_heads_final, decoder_input, 
                        shortcut_type, use_F_sdpa, use_v_pred, 
                        principal_dim, cfg_drop_rate, statistics_path).to(self.device)
        
        self.ema_models = {decay: FlowMatchingModel(data_channels, data_size,
                        patch_size, hidden_size, depth, num_heads,
                        mlp_ratio, attn_drop, proj_drop, num_classes,
                        bottleneck_dim, in_context_len, in_context_start,
                        patch_size_final, hidden_size_final,
                        depth_final, num_heads_final, decoder_input, 
                        shortcut_type, use_F_sdpa, use_v_pred, 
                        principal_dim, cfg_drop_rate, statistics_path).to(self.device) for decay in self.ema_decay_rate}
        
        if use_compile == True:
            self.model.network = torch.compile(self.model.network) 
            for decay in self.ema_decay_rate:
                self.ema_models[decay].network = torch.compile(self.ema_models[decay].network)   
                    
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], gradient_as_bucket_view=True)
        self.optimizer = torch.optim.AdamW(self.model.module.network.parameters(), lr = learning_rate, betas = betas, weight_decay = weight_decay)

        for decay in self.ema_decay_rate:    
            self.ema_models[decay].load_state_dict(self.model.module.state_dict())
            self.ema_models[decay].requires_grad_(False)
            self.ema_models[decay].eval()
                    
        os.makedirs(os.path.join(self.working_dir, 'ckpts'), exist_ok = True)
        ckpt_list = sorted([eval(one.split('.')[0]) for one in os.listdir(os.path.join(self.working_dir, 'ckpts')) if one.endswith('.pt')], reverse = True)
        
        dist.barrier()
        if len(ckpt_list) > 0:
            all = torch.load(os.path.join(self.working_dir, 'ckpts', f'{ckpt_list[0]}.pt'), map_location=self.device)
            self.model.module.load_state_dict({k: all['model'].get(k.replace('_orig_mod.', ''), all['model'].get(k)) for k in self.model.module.state_dict()})
            for decay in self.ema_decay_rate:
                self.ema_models[decay].load_state_dict({
                    k: all['ema_models'][decay].get(k.replace('_orig_mod.', ''), 
                       all['ema_models'][decay].get(k)) for k in self.ema_models[decay].state_dict()
                })
            
            self.optimizer.load_state_dict(all['optimizer'])
            self.finished_steps = all['finished_steps']
            self.finished_epochs = all['finished_epochs']
            if rank == 0:
                self.logger.info(f"Loaded checkpoint from {os.path.join(self.working_dir, 'ckpts', f'{ckpt_list[0]}.pt')}")
                self.logger.info(f"Model Parameters: {sum([p.numel() for p in self.model.parameters()])/1e6:.2f} M")
        else:     
            if rank == 0:
                torch.save({
                    'model':{k.replace('_orig_mod.', ''): v.cpu() for k, v in self.model.module.state_dict().items()},
                    'ema_models':{decay:{k.replace('_orig_mod.', ''): v.cpu() for k, v in self.ema_models[decay].state_dict().items()} for decay in self.ema_decay_rate},
                    'optimizer':self.optimizer.state_dict(),
                    'finished_steps':self.finished_steps,
                    'finished_epochs':self.finished_epochs
                }, os.path.join(self.working_dir, 'ckpts', '0.pt'))
                self.logger.info(f"Saving checkpoint to {os.path.join(self.working_dir, 'ckpts', '0.pt')}")
                self.logger.info(f"Model Parameters: {sum([p.numel() for p in self.model.parameters()])/1e6:.2f} M")

        dist.barrier()
        self.ema_params_list = [list(self.ema_models[decay].parameters()) for decay in self.ema_decay_rate]
        self.model_params_list = list(self.model.module.parameters())
        
        net = self.model.module.network
        decoder = getattr(net, '_orig_mod', net).decoder
        d = decoder.norm.weight.numel()
        m = decoder.adaLN_modulation[-1]

        self.slow_tensors = [m.weight[:-2 * d].detach(), m.bias[:-2 * d].detach()]
        if hasattr(decoder, 'x_in'):
            self.slow_tensors += [p.detach() for layer in (decoder.x_in, decoder.x_out) for p in layer.parameters()]
                
    def train(self):
        
        avg_stats = {'mse': torch.zeros(1).to(self.device),
                     'weight': torch.zeros(1).to(self.device),
                     'grad': torch.zeros(1).to(self.device)}
                        
        start_time = time.time()
        while True:
            self.model.train()
            self.sampler.set_epoch(self.finished_epochs)
                
            for images, labels in self.train_loader:
                
                images = images.to(self.device, non_blocking = True)
                labels = labels.to(self.device, non_blocking = True)
                                            
                batch_size = images.size(0)
                # logit-normal sampling
                t = torch.sigmoid(torch.randn(batch_size, device = self.device) * self.time_std + self.time_mean) 
                
                loss, loss_mse, loss_weight = self.model(images, t, labels, self.use_bf16_autocast)                    
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.finished_steps < self.warmup_steps:
                    curr_lr = self.learning_rate * ((self.finished_steps + 1) / self.warmup_steps)
                else:
                    curr_lr = self.learning_rate
                    
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = curr_lr

                grad = torch.nn.utils.clip_grad_norm_(self.model.module.network.parameters(), max_norm=float('inf'))
                
                slow = (
                    self.shortcut_lrdecay is not None
                    and self.finished_steps >= self.shortcut_lrdecay[0]
                    and self.shortcut_lrdecay[1] != 1.0
                )
                old = [p.clone() for p in self.slow_tensors] if slow else None
                self.optimizer.step()
                if slow:
                    with torch.no_grad():
                        for p, q in zip(self.slow_tensors, old):
                            p.lerp_(q, 1.0 - self.shortcut_lrdecay[1])
                            
                    if self.finished_steps == self.shortcut_lrdecay[0] and dist.get_rank() == 0:
                        self.logger.info(
                            f"Shortcut LR decay enabled after {self.finished_steps} steps: "
                            f"multiplier={self.shortcut_lrdecay[1]:g}, "
                            f"effective_lr={curr_lr * self.shortcut_lrdecay[1]:.6g}"
                        )
        
                with torch.inference_mode():
                
                    for i, decay in enumerate(self.ema_decay_rate):
                        torch._foreach_mul_(self.ema_params_list[i], decay)
                        torch._foreach_add_(self.ema_params_list[i], self.model_params_list, alpha = 1-decay)
                                                
                    avg_stats['mse'] += loss_mse
                    avg_stats['weight'] += loss_weight
                    avg_stats['grad'] += grad
                                            
                    self.finished_steps += 1
                    if self.finished_steps % self.logging_steps == 0:
                        torch.cuda.synchronize()
                        end_time = time.time()
                                                
                        for key in avg_stats.keys():
                            avg_stats[key] /= self.logging_steps
                            dist.all_reduce(avg_stats[key], op = dist.ReduceOp.SUM)
                            avg_stats[key] = avg_stats[key] / dist.get_world_size()
                        
                        if dist.get_rank() == 0:
                            percentage = self.finished_steps / self.total_steps * 100
                            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                            torch.cuda.reset_peak_memory_stats()
                            stats_msg = " | ".join([f"{k}=" + (f"{v.item():.5f}" if v.numel() == 1 else \
                                        "[" + ",".join(f"{x:.3f}" for x in v) + "]") for k, v in avg_stats.items()])
                            self.logger.info(f"Steps = {self.finished_steps:07d}/{self.total_steps:07d} ({percentage:.3f}%) | {stats_msg} |" + \
                                f" lr = {self.optimizer.param_groups[0]['lr']:6f}, Mem = {peak_memory:.2f} GB, Time = {end_time - start_time:.2f} seconds")
                        
                        for key in avg_stats.keys():
                            avg_stats[key] = torch.zeros_like(avg_stats[key])
                        start_time = time.time()
                        
                    if self.finished_steps % self.eval_demo_steps == 0:
                        # Producing visual samples
                        sample_amount = self.eval_demo_shape[0] * self.eval_demo_shape[1]
                        labels = torch.randint(0, self.ema_models[self.first_ema].num_classes, (sample_amount, ), device = self.device)
                        generated = self.ema_models[self.first_ema].inference(labels, self.eval_solver, self.eval_solver_steps, self.eval_cfg_scale, self.eval_cfg_interval, self.use_bf16_autocast)
                                  
                        if dist.get_rank() == 0:
                            os.makedirs(os.path.join(self.working_dir, 'figs'), exist_ok = True)
                            out_path = os.path.join(self.working_dir, 'figs', f'{self.finished_steps}.png')
                            self.visualize_batch(out_path, generated, self.eval_demo_shape[0], self.eval_demo_shape[1])
                            self.logger.info(f"{self.eval_demo_shape[0]} * {self.eval_demo_shape[1]} = {sample_amount} samples ({self.eval_solver}, {self.eval_solver_steps} steps, CFG: {self.eval_cfg_scale}, {self.eval_cfg_interval}) are saved to {out_path}")

                    if self.finished_steps % self.ckpt_saving_steps == 0:
                        if dist.get_rank() == 0:
                            torch.save({
                                'model':{k.replace('_orig_mod.', ''): v.cpu() for k, v in self.model.module.state_dict().items()},
                                'ema_models':{decay:{k.replace('_orig_mod.', ''): v.cpu() for k, v in self.ema_models[decay].state_dict().items()} for decay in self.ema_decay_rate},
                                'optimizer':self.optimizer.state_dict(),
                                'finished_steps':self.finished_steps,
                                'finished_epochs':self.finished_epochs
                            }, os.path.join(self.working_dir, 'ckpts', f'{self.finished_steps}.pt'))
                            self.logger.info(f"Saving checkpoint to {os.path.join(self.working_dir, 'ckpts', f'{self.finished_steps}.pt')}")
                            
                    if self.finished_steps % self.eval_steps == 0:
                                    
                        for ema_param in self.ema_models[self.first_ema].parameters():
                            dist.broadcast(ema_param.data, src = 0)

                        results = [] # Producing FID-50K evaluation npz files
                        world_size = dist.get_world_size()
                        rank = dist.get_rank()
                        generate_amount = (50000 + world_size - 1) // world_size
                        all_labels = torch.arange(world_size * generate_amount, device = self.device) % self.ema_models[self.first_ema].num_classes
                        for start in tqdm.trange(0, generate_amount, self.eval_batch_size, desc = 'Inference for FID-50K Evaluation', disable=(dist.get_rank() != 0)):
                            end = min(generate_amount, start + self.eval_batch_size)
                            labels = all_labels[rank * generate_amount + start: rank * generate_amount + end]
                            generated = self.ema_models[self.first_ema].inference(labels, self.eval_solver, self.eval_solver_steps, self.eval_cfg_scale, self.eval_cfg_interval, self.use_bf16_autocast)
                                      
                            generated_samples = (generated * 127.5 + 127.5).round().clamp(min=0,max=255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
                            if dist.get_rank() == 0:
                                gathered_results_list = [torch.zeros_like(generated_samples) for _ in range(dist.get_world_size())]
                            else:
                                gathered_results_list = None
                            dist.barrier()
                            dist.gather(generated_samples, gather_list=gathered_results_list, dst=0)
                            if dist.get_rank() == 0:
                                results.append(torch.cat(gathered_results_list, dim = 0).cpu())
                                self.logger.info(f"{end}/{generate_amount} samples for FID-50K")
                        torch.cuda.empty_cache()
                        
                        if dist.get_rank() == 0:
                            results = torch.cat(results, dim = 0)[:50000].numpy()
                            os.makedirs(os.path.join(self.working_dir, 'evals'), exist_ok = True)
                            out_path = os.path.join(self.working_dir, 'evals', f'{self.finished_steps}.npz')
                            np.savez(out_path, results)
                            self.logger.info(f"50000 samples ({self.eval_solver}, {self.eval_solver_steps} steps, CFG: {self.eval_cfg_scale}, {self.eval_cfg_interval}) are saved to {out_path}")
                        
                    
                if self.finished_steps >= self.total_steps:
                    dist.destroy_process_group()
                    return 
            self.finished_epochs += 1
            
    def visualize_batch(self, save_name, images, num_rows = 4, num_cols = 8):
        # images: [num_rows * num_cols, data_channels, data_size, data_sizde]
        plt.figure(figsize=(2 * num_cols, 2 * num_rows))
        for i in range(num_rows * num_cols):
            plt.subplot(num_rows, num_cols, i + 1)
            img = (images[i].cpu().permute(1,2,0) * 127.5 + 127.5).round().clamp(min=0,max=255).to(torch.uint8)
            plt.imshow(img, cmap = 'gray' if img.size(-1) == 1 else 'viridis')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required=True, help = 'Training config path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    if int(os.environ['RANK']) == 0: 
        os.makedirs(config['working_dir'], exist_ok = True)
        if not os.path.exists(os.path.join(config['working_dir'],'config.yaml')) or not os.path.samefile(args.config, os.path.join(config['working_dir'],'config.yaml')):
            shutil.copy(args.config, os.path.join(config['working_dir'],'config.yaml'))
        
    learner = Learner(**config)
    learner.train()
    
    