# [NeurIPS 2026 Spotlight] What Kills $v$-Prediction? A Patch-wise PCA Perspective on Pixel-Space Flow Matching

Official PyTorch implementation of **RiT** (NeurIPS 2026 Spotlight, top 1.3% of submissions).

[Tianze Luo](https://github.com/luotianze666), [Haofeng Huang](https://github.com/jason-huang03), [Yuan Yao](https://collegeai.tsinghua.edu.cn/en/People/Full_time_PI/Yuan_Yao.htm)
Tsinghua University

[Paper](https://openreview.net/forum?id=FwWaPhPb2O) | [Checkpoints](https://huggingface.co/luotianze666/RiT_ckpts)

![](./figures/demo.png)
## Code Structure

```text
rit.py                      # Model architecture and decoder variants
models.py                   # Flow-matching objectives and sampling
train.py                    # Distributed training
inference.py                # Distributed inference with pretrained checkpoints
evaluator.py                # Image generation metrics, adapted from ADM
get_stats.py                # Distributed computation of patch statistics
stats.pt                    # Precomputed ImageNet patch statistics
analysis/
├── draw_eigens.py          # Plot the PCA eigenvalue spectrum
├── draw_embeds.py          # Plot embedding effective rank and PCA alignment over training
└── draw_r2.py              # Plot layer-wise linear reconstruction R²
```

## 1. Installation

Run all commands from the repository root. Training and inference require NVIDIA GPUs.

Create the environment for training, inference, and analysis:

```bash
conda create -n rit python=3.10 -y
conda activate rit
pip install -r requirements_rit.txt
pip install -U "huggingface_hub[cli]>=0.34,<1.0"
```

Create a separate environment for evaluation:

```bash
conda create -n rit_eval python=3.10 -y
conda activate rit_eval
pip install -r requirements_rit_eval.txt
```

## 2. Download Checkpoints

Checkpoints and experiment configs are hosted on [Hugging Face](https://huggingface.co/luotianze666/RiT_ckpts). The complete release is approximately **400 GB**; download only the files needed for your experiment.

For example, download the final RiT-B checkpoint, its config, and the ImageNet evaluation reference batch:

```bash
conda activate rit
hf download luotianze666/RiT_ckpts \
    RiT-B-v/config.yaml RiT-B-v/ckpts/750000.pt \
    VIRTUAL_imagenet256_labeled.npz \
    --local-dir ./RiT_ckpts
```

For other models, replace the folder and checkpoint filename accordingly. Always download the accompanying `config.yaml`.

To download the entire release:

```bash
hf download luotianze666/RiT_ckpts --local-dir ./RiT_ckpts
```

The reference batch `VIRTUAL_imagenet256_labeled.npz` is provided by [ADM](https://github.com/openai/guided-diffusion/tree/main/evaluations) and is also available in our checkpoint repository. It is required for evaluation, **not for training or inference**.

## 3. Training

Prepare the ImageNet training images in an `ImageFolder` layout. The released configs use `../imagenet`, alongside the RiT repository:

```text
parent_directory/
├── imagenet/
│   ├── n01440764/
│   │   └── *.JPEG
│   └── ...
└── RiT/
    ├── train.py
    └── your_config.yaml
```

Create `your_config.yaml` using a model's `config.yaml` from [Hugging Face](https://huggingface.co/luotianze666/RiT_ckpts) as a template. Start with the released config for your desired model size, then update `dataset_path` and `working_dir` for your machine. Relative paths are resolved from the repository root.

All supported model variants and training settings are controlled through the config:

- **Backbone:** `hidden_size`, `depth`, and `num_heads` determine the model size. `bottleneck_dim` sets the width of the linear input bottleneck. Use the released B/L/H configs to keep these settings consistent.
- **Decoder and shortcuts:** `depth_final`, `hidden_size_final`, `num_heads_final`, and `patch_size_final` configure the decoder. `decoder_input` controls whether the noisy input is injected into the decoder Transformer blocks, while `shortcut_type` specifies the output-side shortcut formulation, including direct raw-input bypasses and learnable fractional parameterizations.
- **Prediction objective and PCA experiments:** `use_v_pred: True` selects velocity prediction; `False` selects clean-image prediction. Keep `principal_dim: 0` for standard training. A positive value enables PCA-subspace training and requires `statistics_path` to point to `stats.pt`.
- **Optimization:** `batch_size` is the global batch size and must be divisible by the number of GPUs. Set `total_steps`, `learning_rate`, `betas`, `weight_decay`, and `warmup_steps` as needed, or retain the released recipe to reproduce an experiment.

Launch training:

```bash
conda activate rit
torchrun --nproc_per_node=8 train.py --config your_config.yaml
```

Training automatically resumes from the largest numbered checkpoint in `<working_dir>/ckpts/`, restoring model weights, EMA weights, optimizer state, and progress counters. **Use a new, empty `working_dir` to train from scratch.** To extend an existing run, increase `total_steps` and launch the same command.

Checkpoints, sample grids, and evaluation samples are saved under `ckpts/`, `figs/`, and `evals/`. Set `ckpt_saving_steps`, `eval_demo_steps`, and `eval_steps` to control their frequency. Training generates the evaluation samples; run `evaluator.py` separately to compute their metrics.

## 4. Inference

Reference results on ImageNet 256×256, using EMA 0.9996 and 50-step Heun sampling:

| Model | Checkpoint | Approx. epochs | CFG | CFG interval | IS ↑ | FID-50K ↓ |
|---|---:|---:|---:|---|---:|---:|
| RiT-B/16 | 250000 | 200 | 3.1 | [0.0, 0.89] | 283.9 | 3.40 |
| RiT-B/16 | 750000 | 600 | 2.8 | [0.0, 0.89] | 300.3 | 3.01 |
| RiT-L/16 | 250000 | 200 | 2.5 | [0.0, 0.89] | 300.6 | 2.48 |
| RiT-L/16 | 750000 | 600 | 2.6 | [0.0, 0.89] | 331.4 | 2.06 |
| RiT-H/16 | 250000 | 200 | 2.4 | [0.0, 0.89] | 309.2 | 2.17 |
| RiT-H/16 | 600000 | 480 | 2.35 | [0.0, 0.90] | 332.9 | 1.79 |

IS and FID values are rounded to one and two decimal places, respectively.

To generate samples with RiT-B:

```bash
conda activate rit
torchrun --nproc_per_node=8 inference.py \
    --working_dir ./RiT_ckpts/RiT-B-v \
    --ckpt_steps 750000 --ema_decay_rate 0.9996 \
    --eval_solver Heun --eval_solver_steps 50 \
    --eval_cfg_scale 2.8 --eval_cfg_interval 0.0 0.89 \
    --seed 42
```

For RiT-L or RiT-H, download the corresponding config and checkpoint, then use `RiT-L-v` or `RiT-H-v` as the working directory and the sampling settings listed above.

The script reads the architecture from `<working_dir>/config.yaml` and writes 50,000 samples to `<working_dir>/evals/`. Allow approximately **10 GB** of disk space per sample file. Inference does not require the ImageNet training set.

Use `--eval_cfg_scale 1.0` for unguided sampling. `--eval_batch_size` controls the batch size per GPU. BF16 and compilation are enabled by default; disable them with `--no-use_bf16_autocast` and `--no-use_compile`. Run `python inference.py --help` for all options.

## 5. Evaluation

Download `VIRTUAL_imagenet256_labeled.npz` as shown above, then run:

```bash
conda activate rit_eval
python evaluator.py \
    --ref_batch ./RiT_ckpts/VIRTUAL_imagenet256_labeled.npz \
    --sample_batch_dir ./RiT_ckpts
```

`--sample_batch_dir` should point to the **parent directory of the experiments**, not an individual `evals/` folder. The evaluator scans each experiment's `evals/`, computes the metrics, appends them to `eval_results.txt`, and **deletes each NPZ after successful evaluation**.

Previously evaluated filenames are skipped. To evaluate the same filename again, rename the existing `eval_results.txt` or remove the matching entry.

For continuous evaluation:

```bash
conda activate rit_eval
while true; do
    python -u evaluator.py \
        --ref_batch ./RiT_ckpts/VIRTUAL_imagenet256_labeled.npz \
        --sample_batch_dir ./RiT_ckpts
    sleep 60
done
```

This checks for new files 60 seconds after each pass finishes. Run only one evaluator per output directory, preferably on a separate GPU. When evaluating alongside training, ensure sample files are fully written before they become visible to the evaluator.

## 6. Analysis Figures
We provide reproducible scripts for generating the analysis figures presented in our paper directly from the released checkpoints. These scripts cover the patch-wise PCA eigenspectrum, the evolution of learned input embeddings, and layer-wise linear reconstruction fidelity.

The eigenvalue plot only needs the `stats.pt` included in this repository. The remaining figures require checkpoints from Hugging Face:

- **R²:** the configs and 250K checkpoints of `PCA768-v`, `PCA128-v`, `PCA768-x`, and `PCA128-x`, plus the ImageNet training images.
- **Embedding plots:** the full checkpoint trajectories of those four experiments and `vanilla-L-v`.

To download the PCA experiment trajectories and vanilla-L checkpoints:

```bash
conda activate rit
hf download luotianze666/RiT_ckpts \
    --include 'PCA*/config.yaml' 'PCA*/ckpts/*.pt' \
              'vanilla-L-v/config.yaml' 'vanilla-L-v/ckpts/*.pt' \
    --local-dir ./RiT_ckpts
```

Generate the figures:

```bash
python analysis/draw_eigens.py \
    --stats_path ./stats.pt \
    --save_path ./figures/pca_eigenvalues.pdf

python analysis/draw_embeds.py \
    --stats_path ./stats.pt --top_k 280 \
    --save_path ./figures/embedding.pdf

python analysis/draw_r2.py \
    --dataset_path ../imagenet --stats_path ./stats.pt \
    --batch_size 64 --steps 16 --seed 42 \
    --save_path ./figures/r2_scores.pdf
```

The scripts save four figures under `./figures/`; `draw_embeds.py` produces separate rank and alignment plots. Default checkpoint paths follow the download layout above. 

To recompute patch statistics:

```bash
torchrun --nproc_per_node=8 get_stats.py \
    --dataset_path ../imagenet \
    --data_size 256 --patch_size 16 \
    --batch_size 256 --save_path ./stats.pt
```

## Acknowledgements

Our implementation builds on [JiT](https://github.com/LTH14/JiT), [SiT](https://github.com/willisma/SiT), and [LightningDiT](https://github.com/hustvl/LightningDiT). We use the evaluation code from [ADM](https://github.com/openai/guided-diffusion/tree/main/evaluations).

## Citation

```bibtex
@inproceedings{luo2026what,
    title={What kills {$v$}-prediction? A Patch-wise PCA Perspective on Pixel-Space Flow Matching},
    author={Luo, Tianze and Huang, Haofeng and Yao, Yuan},
    booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
    year={2026},
    url={https://openreview.net/forum?id=FwWaPhPb2O}
}
```

## License

Original RiT code is licensed under [MIT](LICENSE). Third-party code retains its original licenses.