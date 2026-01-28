# HyperBandit

[![Paper](https://img.shields.io/badge/CIKM'23-Oral-red)](https://arxiv.org/pdf/2308.08497)
[![Paper](https://img.shields.io/badge/TOIS-red)](https://arxiv.org/pdf/2308.08497)
[![arXiv](https://img.shields.io/badge/arXiv-2308.08497-b31b1b.svg)](https://arxiv.org/pdf/2308.08497)

This repository provides the official PyTorch implementation for the **CIKM 23 Oral** paper **"HyperBandit: Contextual Bandit with Hypernetwork for Time-Varying User Preferences in Streaming Recommendation"** and the **TOIS** paper **"Enhancing Bandit Algorithms with LLMs for Time-varying User Preferences in Streaming Recommendations"**


## Quick Start


### Settings

### `--warm_start`
- **Type:** flag (boolean switch)
- **Usage:** presence = `True`, absence = `False`
- **Example:** `--warm_start`
- **Description:** Enable warm-start training (e.g., initialize from a previous state / pretrained model).
- ✅ **HyperBandit:** set to **False** → **do not add** `--warm_start`.
- 🟦 **HyperBandit+:** set to **True** → **add** `--warm_start`.
---

### `--time_embedding`
- **Type:** string
- **Example:** `--time_embedding="glove"`
- **Description:** Time embedding strategy used to encode temporal information.
- **Options:** `polar` / `onehot` / `learn` / `glove`
- ✅ **HyperBandit:** use `--time_embedding="glove"`.
- 🟦 **HyperBandit+:** use `--time_embedding="polar"`

---

### `--dataset`
- **Type:** string
- **Example:** `--dataset="NYC"`
- **Description:** Dataset identifier; used to select dataset-specific preprocessing and evaluation protocol.
- **Options:** `NYC` / `TKY` / `kuai`

---

### `--feature`
- **Type:** string
- **Example:** `--feature="LLM_with_attribute_pca"` or `--feature="glove_pca"`
- **Description:** Observed feature configuration used as the bandit context representation.
- **Options:** `LLM_with_attribute_pca` / `glove_pca`
- ✅ **HyperBandit:** use `--feature="glove_pca"`.
- 🟦 **HyperBandit+:** use `--feature="LLM_with_attribute_pca"`.

### Run SpecGR for Recommendation Using Preprocessed Data and Checkpoints

```bash
bash scripts/quick_start.sh
```

## Citation

If you find our code or paper helpful, please cite our work:
```bibtex
@inproceedings{shen2023hyperbandit,
  title={Hyperbandit: Contextual bandit with hypernewtork for time-varying user preferences in streaming recommendation},
  author={Shen, Chenglei and Zhang, Xiao and Wei, Wei and Xu, Jun},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={2239--2248},
  year={2023}
}

```
