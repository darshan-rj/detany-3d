# DetAny3D

Promptable 3D detection foundation model for monocular images. This repository contains code for training, inference, and deployment of the DetAny3D system.

---

Recent repository changes detected (automated scan):
- Duplicate README content consolidated into this file.
- `deploy_streamlit_test.py` appears to have been removed from the tree.

If any of these notes are incorrect, please tell me and I will adjust the README accordingly.

## What this repo contains
- `detect_anything/` — model implementation, configs, datasets, and utilities.
- `modeling/` — image encoder, transformer, prompt & mask modules.
- `train.py`, `train_utils.py`, `wrap_model.py` — training and model wrappers.
- `deploy.py`, `deploy_streamlit.py`, `app.py` — demo and deployment scripts.
- `custom_dataset_preparation/` — dataset conversion helpers.
- `data/` — expected dataset files and `DA3D_pkls/` for minimal inference metadata.

## Quickstart (minimal)

1) Create the environment (Windows / PowerShell):

```powershell
conda create -n detany3d python=3.8; conda activate detany3d
```

2) Install key third-party projects (SAM, UniDepth, GroundingDINO) per their instructions. Example for GroundingDINO:

```powershell
git clone https://github.com/IDEA-Research/GroundingDINO.git; cd GroundingDINO; pip install -e .
```

3) Install Python dependencies for this repo:

```powershell
pip install -r requirements.txt
```

4) Place checkpoints in a private folder (example layout):

```
detany3d_private/
└── checkpoints/
    ├── sam_ckpts/sam_vit_h.pth
    ├── unidepth_ckpts/unidepth.pth
    ├── dino_ckpts/dino_swin_large.pth
    └── detany3d_ckpts/detany3d.pth
```

## Data layout (expected)

Follow the Omni3D conventions where applicable. Minimal folders the code expects under `data/`:

```
data/
├── DA3D_pkls/
├── kitti/
├── nuscenes/
├── hypersim/
├── waymo/
├── objectron/
└── SUNRGBD/
```

Note: Depth files are not required for inference. To skip depth loading, set `depth_path = None` in `detect_anything/datasets/detany3d_dataset.py`.

## Common commands

- Train (distributed example — adapt env vars for your cluster):

```powershell
torchrun --nproc_per_node=8 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --nnodes=8 --node_rank=${RANK} ./train.py --config_path ./detect_anything/configs/train.yaml
```

- Run inference (example):

```powershell
torchrun --nproc_per_node=8 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --nnodes=1 --node_rank=${RANK} ./train.py --config_path ./detect_anything/configs/inference_indomain_gt_prompt.yaml
```

After inference you will find `{dataset}_output_results.json` under `exps/<your_exp_dir>/`.

## Launch demo

- Local demo server:

```powershell
python ./deploy.py
```

- Streamlit demo:

```powershell
python ./deploy_streamlit.py
```

## Development notes

- The project integrates external codebases (SAM, GroundingDINO, UniDepth). Follow each project's install instructions to avoid compatibility issues.
- Configurations live in `detect_anything/configs/` — copy and adapt these YAMLs for experiments.
- If you plan to run only inference, use `DA3D_pkls` and avoid large dataset downloads.

## Detected repo maintenance actions you might want

- Commit this README consolidation and push to a feature branch if you want review.
- If `deploy_streamlit_test.py` was intentionally removed but should be preserved, restore it from backup or git history.

If you'd like, I can also:
- create a small `CONTRIBUTING.md`,
- add a short `docs/` page for dataset conversion, or
- create a branch and open a PR with this README change.

---

If this looks good, I will finalize the change (commit/PR) or adapt it further — tell me which next step you prefer.

