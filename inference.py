#!/usr/bin/env python3

import argparse
import gc
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# Constants
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = SCRIPT_DIR / "models" / "qwen25vl"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".gif"}
LETTER_TO_INT = {"A": 1, "B": 2, "C": 3, "D": 4, "X": 5}
INT_TO_LETTER = {v: k for k, v in LETTER_TO_INT.items()}

MAX_NEW_TOKENS = 16   # only need a single letter
MAX_PIXELS = 1280 * 28 * 28   # cap to prevent OOM on tall images

SYSTEM_PROMPT = (
    "You are an expert in deep learning, machine learning, and mathematics.\n"
    "You will be shown an image of a multiple-choice question.\n"
    "The question has exactly FOUR options: (A), (B), (C), and (D).\n\n"
    "Your task:\n"
    "1. Read the question and all four options carefully from the image. "
    "Pay close attention to mathematical notation, Greek letters, and formulas.\n"
    "2. Identify the single correct answer using your deep learning expertise.\n"
    "3. Reply with ONLY a single uppercase letter: A, B, C, or D. "
    "If you are genuinely unsure, reply with X.\n\n"
    "Do NOT include any explanation, punctuation, or extra text — just the single letter."
)
USER_PROMPT = (
    "Look at the multiple-choice question in the image above.\n"
    "What is the correct answer? Reply with only A, B, C, or D.\n"
    "If genuinely unsure, reply with X."
)


# Helpers
def parse_answer(raw: str) -> str:
    """Extract A/B/C/D/X from raw model output."""
    if raw is None:
        return "X"
    text = raw.strip().upper()
    if text in ("A", "B", "C", "D", "X"):
        return text
    m = re.search(r"(?:ANSWER(?:\s+IS)?[:\s]*|\()\s*([A-D])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"[A-D]", text)
    if m:
        return m.group()
    return "X"


def find_images_dir(test_dir: Path) -> Path:
    """Locate the images directory inside test_dir."""
    common_names = ["images", "test_images", "imgs", "image", "data", "test"]
    for name in common_names:
        cand = test_dir / name
        if cand.is_dir():
            # verify it actually contains images
            try:
                if any(p.suffix.lower() in IMG_EXTS for p in cand.iterdir()):
                    return cand
            except Exception:
                pass
    # Look for any subdirectory containing images
    for sub in test_dir.iterdir():
        if sub.is_dir():
            try:
                if any(p.suffix.lower() in IMG_EXTS for p in sub.iterdir()):
                    return sub
            except Exception:
                continue
    # Fallback: use test_dir itself
    if any(p.suffix.lower() in IMG_EXTS for p in test_dir.iterdir() if p.is_file()):
        return test_dir
    raise FileNotFoundError(
        f"Could not locate any images directory inside {test_dir}"
    )


def find_image(images_dir: Path, name: str) -> Path:
    """Locate an image file given a (possibly extension-less) name."""
    name = str(name).strip()
    # Direct match
    direct = images_dir / name
    if direct.is_file():
        return direct
    # Try adding common extensions
    stem_path = images_dir / name
    for ext in IMG_EXTS:
        cand = Path(str(stem_path) + ext)
        if cand.is_file():
            return cand
    # Try treating name as having an extension we should strip and re-add
    stem = Path(name).stem
    for ext in IMG_EXTS:
        cand = images_dir / (stem + ext)
        if cand.is_file():
            return cand
    # Case-insensitive search through directory
    name_lower = name.lower()
    stem_lower = stem.lower()
    for f in images_dir.iterdir():
        if not f.is_file():
            continue
        if f.name.lower() == name_lower or f.stem.lower() == stem_lower:
            return f
    return direct  # caller can check exists()


def detect_id_column(df: pd.DataFrame) -> str:
    """Find the column most likely to hold image identifiers."""
    candidates = {"image_id", "image", "filename", "file_name", "id",
                  "image_name", "name", "img", "img_id", "img_name", "image_path"}
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return df.columns[0]


def detect_answer_column(template: pd.DataFrame, id_col: str) -> str:
    """Find the column most likely to hold the answer."""
    candidates = {"answer", "label", "prediction", "class", "target",
                  "y", "pred", "predicted", "result", "ans"}
    for c in template.columns:
        if c.lower() in candidates:
            return c
    # Fall back: any column that is not the id column
    for c in template.columns:
        if c != id_col:
            return c
    return template.columns[-1]


def detect_output_letters(template: pd.DataFrame, answer_col: str) -> bool:
    """Decide whether to output letters (A/B/C/D) or integers (1..5)."""
    if answer_col not in template.columns:
        return False
    series = template[answer_col].dropna()
    if len(series) == 0:
        return False
    sample = series.iloc[0]
    if isinstance(sample, str):
        s = sample.strip().upper()
        if s in ("A", "B", "C", "D", "X"):
            return True
    return False


# Main
def main():
    parser = argparse.ArgumentParser(description="GNR project inference")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Absolute path to the test directory")
    args = parser.parse_args()

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        print(f"[ERROR] test_dir does not exist or is not a directory: {test_dir}",
              file=sys.stderr)
        sys.exit(1)

    test_csv_path = test_dir / "test.csv"
    sub_csv_path = test_dir / "submission.csv"

    if not test_csv_path.is_file():
        print(f"[ERROR] test.csv not found in {test_dir}", file=sys.stderr)
        sys.exit(1)

    images_dir = find_images_dir(test_dir)
    print(f"[INFO] Images directory: {images_dir}")

    # Load CSVs
    test_df = pd.read_csv(test_csv_path)
    print(f"[INFO] test.csv : {len(test_df)} rows | columns: {list(test_df.columns)}")

    if sub_csv_path.is_file():
        sub_template = pd.read_csv(sub_csv_path)
        print(f"[INFO] submission.csv template : {len(sub_template)} rows | "
              f"columns: {list(sub_template.columns)}")
    else:
        sub_template = None
        print("[WARN] No submission.csv template found; creating one from test.csv structure.")

    id_col = detect_id_column(test_df)
    print(f"[INFO] Using id column: {id_col!r}")

    if sub_template is not None:
        answer_col = detect_answer_column(sub_template, id_col)
    else:
        answer_col = "answer"
    print(f"[INFO] Using answer column: {answer_col!r}")

    output_letters = (
        detect_output_letters(sub_template, answer_col)
        if sub_template is not None else False
    )
    print(f"[INFO] Output format: {'LETTERS (A/B/C/D)' if output_letters else 'INTEGERS (1..5)'}")

    # Load model
    print(f"[INFO] Loading processor & model from {MODEL_PATH} ...")
    if not MODEL_PATH.is_dir():
        print(f"[ERROR] Model directory not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)

    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        max_pixels=MAX_PIXELS,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("[INFO] Model loaded.")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"[INFO] GPU {i}: {used:.1f} / {total:.1f} GB used")

    first_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Inference function
    @torch.inference_mode()
    def predict(image_path: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(first_device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        gen_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        raw = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        del inputs, output_ids, gen_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return parse_answer(raw)

    # Iterate over the test set
    predictions = []  # parallel to test_df rows

    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       desc="Predicting", unit="img"):
        img_name = str(row[id_col])
        img_path = find_image(images_dir, img_name)
        if not img_path.is_file():
            print(f"[WARN] Image not found for id={img_name!r} (looked in {images_dir})")
            predictions.append("A")
            continue
        try:
            letter = predict(str(img_path))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            try:
                letter = predict(str(img_path))
            except Exception as e:
                print(f"[ERROR] OOM retry failed for {img_name}: {e}")
                letter = "A"
        except Exception as e:
            print(f"[ERROR] Inference failed for {img_name}: {e}")
            letter = "A"
        predictions.append(letter)

    # Build submission
    # Map letter predictions to integers 1-5:
    #   A=1, B=2, C=3, D=4, X=5  (5 means model was genuinely unsure)
    # Fallback to 1 only for truly unexpected outputs (should never happen).
    out_values = [LETTER_TO_INT.get(p, 1) for p in predictions]

    # Fixed output schema required by grader: image_id, image_name, option
    # image_id == image_name (both carry the image filename/id from test.csv)
    image_ids = [str(test_df[id_col].iloc[i]) for i in range(len(test_df))]

    sub_out = pd.DataFrame({
        "image_id":   image_ids,
        "image_name": image_ids,
        "option":     out_values,
    })

    out_path = Path("submission.csv").resolve()
    sub_out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path} ({len(sub_out)} rows)")

if __name__ == "__main__":
    main()
