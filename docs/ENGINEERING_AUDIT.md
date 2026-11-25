# SVL-DRL engineering audit

This is an incremental hardening effort. The original training algorithm must be
validated against a baseline before changing its reward, policy, value or
optimization semantics. No benchmark or medical validation is claimed.

## Confirmed issues / follow-ups

1. The original `main_SVL.py` forcibly set `resume_ckpt=True` and
   `save_checkpoint=True`. Both flags are now opt-in.
2. Dataset paths are hardcoded in `main_SVL.py` and `utils/dataset_new.py`.
   They must be moved to a validated dataset configuration in a later change.
3. `MedicalRewardEvaluator._count_connected_components` does not count
   connected components; it thresholds the number of foreground voxels.
   Replace only after defining expected connectivity and per-class behavior.
4. `MedicalImageTransitionBuffer._select_actions` applies softmax to
   `action_probs`; the training loss later uses `log(action_probs.gather(...))`.
   Verify whether the model emits logits or probabilities, and make sampling
   and loss mathematically consistent before modifying either.
5. `trainer_v9.py` manually all-reduces gradients even though the main entry
   uses `DistributedDataParallel`; this needs a distributed integration test
   before removing the redundant reduction.
6. Checkpoint saving lacks scaler and RNG state. Full resume must restore
   model, optimizer, scheduler, scaler, epoch and RNG states.
7. Training/test scripts load checkpoints via `torch.load`; use trusted
   checkpoints and prefer `weights_only=True` where compatible.
8. Add CPU smoke tests, GPU integration tests and multi-GPU validation.
9. The original repository includes compiled Python cache files; remove
   tracked cache files separately and prevent future additions via gitignore.

## First incremental change

- Introduce `utils/reproducibility.py` for Python, NumPy and PyTorch seeding.
- Add `--seed` and `--deterministic` to the training entry.
- Preserve explicit checkpoint/save flags and fail early on missing pretrained weights.
- Add basic reproducibility tests and a `.gitignore`.

## Validation required before merging

- `python -m pytest tests/test_reproducibility.py`
- One-epoch training smoke test without pretrained checkpoint.
- One-epoch training with a trusted pretrained checkpoint.
- Single-GPU vs multi-GPU numerical comparison.
- Fixed-seed baseline for segmentation, policy and value losses.
