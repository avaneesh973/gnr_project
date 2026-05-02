"""
inference.py  —  Deep Learning MCQ Solver (Vision + Logit Confidence)
Usage:
    python inference.py --test_dir <absolute_path_to_test_dir>

Expected test_dir structure:
    <test_dir>/
        images/          <- folder of MCQ images
        test.csv         <- CSV with at least an 'image_name' column

Output:
    ./submission.csv     <- written in the CURRENT WORKING DIRECTORY
"""

import argparse
import csv
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ─── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="MCQ Solver Inference")
parser.add_argument("--test_dir", type=str, required=True)
parser.add_argument("--mock", action="store_true",
                    help="Skip model load and return dummy answers (for pipeline testing)")
args = parser.parse_args()

TEST_DIR   = Path(args.test_dir)
IMAGE_DIR  = TEST_DIR / "images"
TEST_CSV   = TEST_DIR / "test.csv"
OUTPUT_CSV = Path("./submission.csv")

MODEL_PATH = os.environ.get("MODEL_PATH", "./qwen25vl")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

# ─── PROMPT ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in deep learning and neural networks.
You will be given an image of a multiple-choice question about deep learning.

The question has exactly FOUR options numbered 1, 2, 3, and 4.

Your task:
- Carefully read the question and all four options in the image.
- Identify the single correct answer.
- Reply with ONLY a single digit: 1, 2, 3, or 4.
- Do NOT include any explanation, punctuation, or extra text."""

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

if not args.mock:
    print(f"Loading vision model from {MODEL_PATH} ...")
    gc.collect()
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    print("Model loaded.\n")

    # Get token IDs for "1", "2", "3", "4"
    ANSWER_TOKEN_IDS = [
        processor.tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, 5)
    ]
    print(f"Answer token IDs: { {i+1: tid for i, tid in enumerate(ANSWER_TOKEN_IDS)} }")

# ─── INFERENCE ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict_answer(image_path: str) -> tuple:
    """
    Returns (answer: int [1-4], confidence: float, all_probs: dict).

    Instead of sampling a text output, we:
    1. Do a single forward pass with the image + prompt
    2. Look at the logits of the VERY FIRST generated token
    3. Extract probabilities only for tokens '1','2','3','4'
    4. Softmax over just those 4 -> clean probability distribution
    5. Pick argmax -> the most likely answer with a real confidence score
    """
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": SYSTEM_PROMPT},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # Forward pass — generate exactly 1 new token, then read logits
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,       # gives us logits at each generation step
    )

    # scores[0] = logits at the first generated token, shape [1, vocab_size]
    first_token_logits = outputs.scores[0][0]   # shape: [vocab_size]

    # Extract logits only for tokens '1','2','3','4'
    answer_logits = torch.stack([
        first_token_logits[tid] for tid in ANSWER_TOKEN_IDS
    ])  # shape: [4]

    # Softmax over just these 4 options -> proper probability distribution
    answer_probs = F.softmax(answer_logits, dim=0)

    best_idx    = answer_probs.argmax().item()   # 0-indexed
    answer      = best_idx + 1                   # 1-indexed
    confidence  = answer_probs[best_idx].item()

    all_probs = {i+1: round(answer_probs[i].item(), 4) for i in range(4)}

    return answer, confidence, all_probs


def mock_predict(image_path: str) -> tuple:
    """Returns a dummy answer for pipeline testing without loading any model."""
    return 1, 1.0, {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}

# ─── READ TEST CSV ────────────────────────────────────────────────────────────

if not TEST_CSV.exists():
    raise FileNotFoundError(f"test.csv not found at {TEST_CSV}")

image_names = []
with open(TEST_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row.get("image_name") or row.get("id") or list(row.values())[0]
        image_names.append(name.strip())

print(f"Found {len(image_names)} entries in test.csv")

# ─── BATCH INFERENCE ─────────────────────────────────────────────────────────

results     = []   # list of (stem, answer, confidence, all_probs)
start_time  = time.time()

LOW_CONF_THRESHOLD = 0.50   # flag answers below this confidence

for name in tqdm(image_names, desc="Predicting", unit="img"):
    # Resolve image path (name may or may not have extension)
    img_path = None
    candidate = IMAGE_DIR / name
    if candidate.exists():
        img_path = candidate
    else:
        for ext in IMG_EXTS:
            p = IMAGE_DIR / (name + ext)
            if p.exists():
                img_path = p
                break

    stem = Path(name).stem if '.' in name else name  # strip ext for CSV

    if img_path is None:
        print(f"  Image not found for '{name}' -> defaulting to 1")
        results.append((stem, 1, 0.0, {}))
        continue

    try:
        if args.mock:
            answer, confidence, all_probs = mock_predict(str(img_path))
        else:
            answer, confidence, all_probs = predict_answer(str(img_path))
    except Exception as e:
        print(f"  Error on {name}: {e} -> defaulting to 1")
        results.append((stem, 1, 0.0, {}))
        continue

    if confidence < LOW_CONF_THRESHOLD:
        print(f"  LOW CONF {stem}: answer={answer}, probs={all_probs}")

    results.append((stem, answer, confidence, all_probs))

elapsed = time.time() - start_time
print(f"\nDone in {elapsed/60:.1f} min  ({elapsed/len(image_names):.1f} s/img)")

# ─── CONFIDENCE SUMMARY ───────────────────────────────────────────────────────

confidences = [r[2] for r in results]
if confidences:
    avg_conf  = sum(confidences) / len(confidences)
    low_count = sum(1 for c in confidences if c < LOW_CONF_THRESHOLD)
    print(f"\nConfidence summary:")
    print(f"  Average confidence : {avg_conf:.3f}")
    print(f"  Low-conf (<{LOW_CONF_THRESHOLD}) count : {low_count} / {len(results)}")

# ─── WRITE submission.csv ─────────────────────────────────────────────────────

with open(OUTPUT_CSV, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "answer"])
    for stem, answer, _, _ in results:
        writer.writerow([stem, answer])

print(f"\nSubmission saved -> {OUTPUT_CSV.resolve()}")
