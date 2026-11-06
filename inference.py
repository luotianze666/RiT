import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import numpy as np
from models import FlowMatchingModel
import tqdm
import torch.distributed as dist
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, required=True)
    parser.add_argument('--ckpt_steps', type=int, nargs='+', default=[250000])
    parser.add_argument('--use_compile', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_bf16_autocast', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ema_decay_rate', type=float, nargs='+', default=[0.9996])
    parser.add_argument('--eval_batch_size', type=int, default=250)
    parser.add_argument('--eval_solver_steps', type=int, default=50)
    parser.add_argument('--eval_solver', choices=['Euler', 'Heun'], default='Heun')
    parser.add_argument('--eval_cfg_scale', type=float, nargs='+', default=[1.0])
    parser.add_argument('--eval_cfg_interval', type=float, nargs=2, action='append', default=None)
    parser.add_argument('--seed', type=int, default=42)
    config = vars(parser.parse_args())
    if config['eval_cfg_interval'] is None:
        config['eval_cfg_interval'] = [[0.0, 1.0]]

    with open(os.path.join(config['working_dir'], 'config.yaml')) as file:
        model_config = yaml.safe_load(file)
            
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl', init_method='env://', rank = rank, world_size = world_size, device_id = device)
    torch.cuda.set_device(device)

    torch.backends.cuda.enable_flash_sdp(False)  
    torch.backends.cuda.enable_mem_efficient_sdp(True) 
    torch.backends.cuda.enable_math_sdp(False) 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ema_model = FlowMatchingModel(data_channels = model_config['data_channels'], 
                                  data_size = model_config['data_size'],
                                  patch_size = model_config['patch_size'],
                                  hidden_size = model_config['hidden_size'],
                                  depth = model_config['depth'],
                                  num_heads = model_config['num_heads'],
                                  mlp_ratio = model_config['mlp_ratio'],
                                  attn_drop = model_config['attn_drop'],
                                  proj_drop = model_config['proj_drop'],
                                  num_classes = model_config['num_classes'],
                                  bottleneck_dim = model_config['bottleneck_dim'],
                                  in_context_len = model_config['in_context_len'],
                                  in_context_start = model_config['in_context_start'],
                                  patch_size_final = model_config['patch_size_final'],
                                  hidden_size_final = model_config['hidden_size_final'],
                                  depth_final = model_config['depth_final'],
                                  num_heads_final = model_config['num_heads_final'],
                                  decoder_input = model_config['decoder_input'],
                                  shortcut_type = model_config['shortcut_type'],
                                  use_F_sdpa = model_config['use_F_sdpa'],
                                  use_v_pred = model_config['use_v_pred'],
                                  principal_dim = model_config['principal_dim'],
                                  cfg_drop_rate = model_config['cfg_drop_rate'],
                                  statistics_path = model_config['statistics_path']).to(device)
    
    if config['use_compile'] == True:
        ema_model.network = torch.compile(ema_model.network)
    
    for ckpt_step in config['ckpt_steps']:
        all = torch.load(os.path.join(config['working_dir'], 'ckpts', f"{ckpt_step}.pt"), map_location=device)
        for decay in config['ema_decay_rate']:        
            
            ema_model.load_state_dict({
                    k: all['ema_models'][decay].get(k.replace('_orig_mod.', ''), 
                    all['ema_models'][decay].get(k)) for k in ema_model.state_dict()
                })
            ema_model.requires_grad_(False)
            ema_model.eval()
            
            for cfg_scale in config['eval_cfg_scale']:
                for cfg_interval in config['eval_cfg_interval']:
                    torch.manual_seed(config['seed'] + rank)
                    results = [] # Producing FID-50K evaluation npz files
                    generate_amount = (50000 + world_size - 1) // world_size
                    all_labels = torch.arange(world_size * generate_amount, device = device) % ema_model.num_classes
                    
                    for start in tqdm.trange(0, generate_amount, config['eval_batch_size'], desc = 'Inference for FID-50K Evaluation', disable=(dist.get_rank() != 0)):
                        end = min(generate_amount, start + config['eval_batch_size'])
                        labels = all_labels[rank * generate_amount + start: rank * generate_amount + end]
             
                        generated = ema_model.inference(labels, config['eval_solver'], config['eval_solver_steps'], 
                                                        cfg_scale, cfg_interval, config['use_bf16_autocast'])
                                    
                        generated_samples = (generated * 127.5 + 127.5).round().clamp(min=0,max=255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
                        if dist.get_rank() == 0:
                            gathered_results_list = [torch.zeros_like(generated_samples) for _ in range(dist.get_world_size())]
                        else:
                            gathered_results_list = None
                            
                        dist.barrier()
                        dist.gather(generated_samples, gather_list=gathered_results_list, dst=0)
                        if dist.get_rank() == 0:
                            results.append(torch.cat(gathered_results_list, dim = 0).cpu())

                    torch.cuda.empty_cache()
            
                    if dist.get_rank() == 0:
                        results = torch.cat(results, dim = 0)[:50000].numpy()
                        os.makedirs(os.path.join(config['working_dir'], 'evals'), exist_ok = True)
                        out_path = os.path.join(config['working_dir'], 'evals', f"{ckpt_step}_{decay}_{config['eval_solver']}_{config['eval_solver_steps']}_{cfg_scale}_{cfg_interval}_{config['seed']}.npz")
                        np.savez(out_path, results)
                        print(f"50000 samples are saved to {out_path}")
                