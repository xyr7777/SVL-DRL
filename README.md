# SVL-DRL

Research code for staged voxel-level deep reinforcement learning for 3D medical image segmentation with noisy annotations.

## Reproducibility status

This is a research-code release, not a verified one-command reproduction package. The model modules and the test-data loader are included with the training and evaluation scripts. The exact package versions, dataset split manifests, noise-generation settings, checkpoints, and commands for all experiments have not yet been consolidated and verified. The published scripts contain machine-specific paths and historical defaults. A successful run and numerical agreement with the paper have not been independently verified for this release.

## Repository overview

- `main_SVL.py`: staged training entry point.
- `trainer_v9.py`: staged training and validation loop.
- `swin_unetr_policy_value.py`, `policy_value_v7.py`: model definitions referenced by the training entry point.
- `main_noise.py`, `trainer_noise.py`, `utils/dataset_noise.py`: noise-related code; inspect the selected data loader and noise construction for the experiment being reproduced.
- `test.py`, `data/pre_dataset.py`: evaluation script and its data loader. Review the model and checkpoint used by the script before interpreting it as an SVL-DRL evaluation.
- `utils/dataset_new.py`: dataset loading and transformations used by `main_SVL.py`.

## Data

Datasets must be obtained from their respective providers and are subject to their access conditions:

- LA 2018 Atrial Segmentation Challenge: https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/
- Pancreas-CT: https://www.cancerimagingarchive.net/collection/pancreas-ct/
- BraTS 2021: https://www.med.upenn.edu/cbica/brats2021/

The manuscript reports patient-level splits: LA 80 training / 20 test cases; Pancreas-CT 62 / 20; BraTS 2021 1,000 / 251. Case identifiers for these splits and experiment-specific noise manifests are not included in this release; the counts alone do not determine the splits. Prepare the expected dataset layout and manifests by inspecting the relevant loader before running.

## Environment and setup

The code uses Python, PyTorch, MONAI, NumPy, NiBabel, TorchIO and additional packages imported by the scripts. CUDA is required by the current training entry point. The exact versions used for the reported experiments are not specified in this release; install compatible versions only after checking the original environment. The local research environment included a MONAI source tree; compatibility with an arbitrary MONAI release has not been checked.

After installing the dependencies, inspect available arguments with `python main_SVL.py --help` and `python test.py --help`. These commands require imports to succeed; they do not validate the full training or testing pipeline.

## Training and evaluation notes

1. Obtain and preprocess the data, prepare the exact patient-level split and noisy-label files, and inspect `utils/dataset_new.py` for the selected dataset.
2. Configure dataset paths and checkpoint paths in `main_SVL.py`. Several `data_name` branches overwrite `--data_dir` in code, and `main()` sets `resume_ckpt` to true; passing a path alone may therefore not suffice. A compatible starting checkpoint is needed unless this behavior is changed deliberately.
3. Set the actual experimental parameters explicitly and record the command, software versions, split manifest, random seed, and checkpoint. Script defaults are historical and do not by themselves reproduce the manuscript settings: `optim_name` defaults to AdamW and `max_epochs` to 15000, while the manuscript reports SGD and 1000 epochs. Confirm the settings against the original experiment records before drawing conclusions.
4. The authors report that validation data were used for model selection and that the test cases were independent and not used for model selection. `trainer_v9.py` computes validation metrics during training and saves best checkpoints based on them. The exact split manifests must still be provided to reproduce this separation.
5. `test.py` uses `data/pre_dataset.py` and instantiates its own segmentation model; verify its architecture and checkpoint compatibility before use. Full-volume predictions require the appropriate sliding-window configuration and preprocessing.

The exact commands and additional artifacts needed for an independently verified reproduction are to be added when the original experiment configuration is reconciled with this release.
