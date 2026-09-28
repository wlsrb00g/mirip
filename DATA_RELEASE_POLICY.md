# MIRIP ranking-pilot data and result release policy

This repository does not contain the pairwise ranking pilot's training/validation rows, original artwork, applicant metadata, embeddings, model weights, or per-example predictions.

The local ranking-pilot source references 40,000 training pairs and 4,940 validation pairs. The pair CSVs include references into an image corpus; the source artworks and related applicant records have not been cleared for redistribution. Keep those files local and out of GitHub. `.gitignore` excludes dataset/cache/checkpoint/log locations and common model artifact formats as a second guard; it is not a substitute for review of `git status` before publication.

No 86% ranking-accuracy artifact has been located. The existing evaluator defines pairwise accuracy, but a metric implementation and a dataset's pair counts are not evidence of an observed score. Do not add an 86% claim unless a saved evaluation report identifies the checkpoint, exact held-out split, correct/total pair counts, and evaluation code/version.

For a future public release, publish only (a) code/configuration with redistribution rights confirmed, (b) an aggregate evaluation report with split and checkpoint provenance, and (c) a data card explaining how authorized users can obtain data. Do not publish original artworks, image paths, applicant-level labels, pair rows, embeddings, checkpoints, or raw prediction logs without explicit rights/privacy clearance.

Current execution blocker: the available Python runtime is CPU-only; XPU and CUDA are unavailable. The upstream training CLI accepts only `cpu` or `cuda`, and the local DINOv2-large pretrained weights and a trained ranking checkpoint were not found. No training or external model download has been started. A CPU run would therefore be a protocol/device change and could take substantially longer than a quick evaluation.
