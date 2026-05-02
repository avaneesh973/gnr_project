# GNR Project — Deep Learning MCQ Solver

Vision–language pipeline that reads multiple-choice question images and predicts
the correct option (A / B / C / D). Built on **Qwen/Qwen2.5-VL-7B-Instruct**.

---

## 1. Repository Contents

| File              | Purpose                                                         |
|-------------------|-----------------------------------------------------------------|
| `inference.py`    | Offline inference script. Reads test data, writes predictions. |
| `README.md`       | This file.                                                      |

After `setup.bash` runs, two additional artefacts are created in the working dir:

| Path                  | Description                                  |
|-----------------------|----------------------------------------------|
| `models/qwen25vl/`    | Qwen2.5-VL-7B-Instruct weights (~16 GB).    |
| (conda env)           | `gnr_project_env` (Python 3.11).            |

---

## 2. Target System

| Component  | Specification                              |
|------------|--------------------------------------------|
| OS         | Linux                                      |
| GPU        | NVIDIA L40s (48 GB VRAM, SM_89, CUDA 12.6) |
| RAM        | 16 GB                                      |
| Python     | 3.11                                       |
| Conda env  | `gnr_project_env`                          |

The script is also tested to run on any single CUDA-12.x GPU with ≥ 24 GB VRAM
(the model needs ~16 GB in bf16).

---

## 3. Setup — `bash setup.bash`

> **Internet is required during setup.** It is *not* required during inference.

`setup.bash` performs four steps in order:

1. **Clones the project repository** (this repo) into the current directory.
2. **Creates the conda environment** `gnr_project_env` with Python 3.11.
3. **Installs all Python dependencies** (PyTorch 2.5.1 + cu124 wheels,
   `transformers ≥ 4.49`, `accelerate`, `qwen-vl-utils`, `Pillow`, `pandas`,
   `tqdm`, `huggingface_hub`, `sentencepiece`, etc.).
4. **Downloads the Qwen2.5-VL-7B-Instruct model weights** (~16 GB) from
   HuggingFace into `./models/qwen25vl/`.

### Exact command sequence the grader will run

```bash
cd ./your_directory          # directory created by unzipping the submission
bash setup.bash              # all internet operations happen here
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
python <grading_script> --submission_file submission.csv
conda remove --name gnr_project_env --all -y
```

`setup.bash` uses `set -euo pipefail`, so it aborts immediately on any failure
and prints the offending step.

### Verifying setup succeeded

After `bash setup.bash` finishes you should see:

```
================================================================
  Setup complete.
  Run with:
      conda activate gnr_project_env
      python inference.py --test_dir <absolute_path_to_test_dir>
================================================================
```

and

```bash
$ ls models/qwen25vl/
config.json
generation_config.json
preprocessor_config.json
tokenizer.json
tokenizer_config.json
model-00001-of-000XX.safetensors
...
```

---

## 4. Inference — `python inference.py --test_dir <dir>`

> **No internet is required.** All model weights are already on disk.

### Input — structure of `<test_dir>`

The grader will pass an **absolute path** to a directory laid out exactly like
the sample test set:

```
<test_dir>/
├── test.csv              # list of test items
├── submission.csv        # dummy template — defines the output format
└── images/               # directory of MCQ images
    ├── image_1.png
    ├── image_2.png
    └── ...
```

The script auto-detects:

- The **images sub-directory** (tries `images/`, `test_images/`, `imgs/`,
  any other subdir containing image files; falls back to `<test_dir>` itself).
- The **identifier column** in `test.csv` (`image_id`, `image`, `filename`,
  `id`, etc.).
- The **answer column** in `submission.csv` (`answer`, `label`, `prediction`,
  etc.).
- The **output format** — letters (`A/B/C/D`) or integers (`1/2/3/4`) —
  by inspecting the dummy values in the provided `submission.csv`.

### Output — `submission.csv`

Written to the **current working directory** (i.e. your project directory),
**not** inside `<test_dir>`. The format mirrors the dummy `submission.csv`
exactly: same column names, same row order, same id values; only the answer
column is overwritten with predictions.

### Example

```bash
$ conda activate gnr_project_env
$ python inference.py --test_dir /abs/path/to/sample_test
```

---

## 5. Method Summary

- **Model**: Qwen/Qwen2.5-VL-7B-Instruct (BF16, single-shot multimodal LLM).
- **Resolution cap**: `max_pixels = 1280 × 28 × 28 ≈ 1 MP`. Images larger than
  this are downsampled by the processor to prevent VRAM blow-ups on tall /
  high-resolution document scans.
- **Decoding**: greedy (`do_sample=False`, `max_new_tokens=16`) — deterministic
  output, fastest possible decoding for a single-letter response.
- **Prompt**: a system prompt instructs the model to answer with ONLY a single
  uppercase letter (A / B / C / D), or `X` if it is genuinely unsure.
- **Robustness**:
  - Per-image `try / except` — any failure (corrupt image, OOM, parse error)
    falls back to a default prediction of `A` rather than crashing the run.
  - On CUDA OOM the script clears the cache and retries the image once.
  - The output parser handles `"A"`, `"(A)"`, `"The answer is A."`, and other
    common formats; falls back to `X` only if no letter is found.

---

## 6. Dependencies (installed by `setup.bash`)

| Package            | Version constraint            |
|--------------------|-------------------------------|
| `torch`            | `==2.5.1+cu124`               |
| `torchvision`      | `==0.20.1+cu124`              |
| `transformers`     | `>=4.49.0,<4.55.0`            |
| `accelerate`       | `>=1.0.0`                     |
| `qwen-vl-utils`    | `>=0.0.8`                     |
| `Pillow`           | `>=10.0.0`                    |
| `pandas`           | `>=2.0.0`                     |
| `numpy`            | `<2.0`                        |
| `tqdm`             | `>=4.65.0`                    |
| `huggingface_hub`  | `>=0.24.0`                    |
| `sentencepiece`    | latest                        |
| `protobuf`         | latest                        |
| `av`               | latest                        |
| `einops`           | latest                        |

PyTorch is installed from the cu124 index
(`https://download.pytorch.org/whl/cu124`); cu124 wheels are forward-compatible
with the L40s' CUDA 12.6 driver and support the SM_89 architecture.

---

## 7. Troubleshooting

| Symptom                                              | Likely cause / fix                                                    |
|------------------------------------------------------|-----------------------------------------------------------------------|
| `[ERROR] inference.py not found after cloning!`      | The `REPO_URL` in `setup.bash` is wrong, or the repo isn't public.    |
| `huggingface_hub ... 401 / 403`                      | HuggingFace rate-limit. Re-run `setup.bash`; weights are public.      |
| `CUDA out of memory` during inference                | Reduce `MAX_PIXELS` in `inference.py` (e.g. to `1024 * 28 * 28`).     |
| `submission.csv has wrong column names`              | Should never happen — script copies columns from the dummy template.  |
| `ModuleNotFoundError: qwen_vl_utils`                 | The conda env was not activated before running `inference.py`.        |

---

## 8. Reproducing Predictions Locally

```bash
git clone <this repo URL>
cd <repo name>
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir /absolute/path/to/test_dir
cat submission.csv
```

The setup script is idempotent — re-running it will reuse an existing
`gnr_project_env` and re-validate the model directory.
