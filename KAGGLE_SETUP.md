# Kaggle API Setup Guide

## Quick Setup (3 Steps)

### 1. Get Your Kaggle API Token

1. Go to **https://www.kaggle.com/settings**
2. Scroll down to the **"API"** section
3. Click **"Create New Token"**
4. This downloads `kaggle.json` to your Downloads folder

### 2. Install the Token

```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move the token file (adjust path if needed)
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set proper permissions (security requirement)
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download the Dataset

```bash
# Download and organize ISPRS Potsdam dataset (~20GB)
uv run python scripts/download_isprs_potsdam.py
```

That's it! The script will:
- ✓ Download the dataset from Kaggle
- ✓ Extract all archives
- ✓ Organize files into `data/isprs_potsdam/images/` and `data/isprs_potsdam/labels/`
- ✓ Clean up temporary files
- ✓ Verify the dataset is ready

---

## Troubleshooting

### "401 Unauthorized" Error
Your kaggle.json is not in the right place or has wrong permissions.

**Fix:**
```bash
# Make sure it's in ~/.kaggle/
ls -la ~/.kaggle/kaggle.json

# Should show: -rw------- (permissions 600)
# If not, run:
chmod 600 ~/.kaggle/kaggle.json
```

### "kaggle: command not found"
The Kaggle package is already installed via `uv add kaggle`.

**Fix:**
```bash
# Use the download script which handles this
uv run python scripts/download_isprs_potsdam.py
```

### Download is Very Slow
The dataset is ~20GB. On slow connections this can take 30+ minutes.

**Alternative:**
```bash
# Try a smaller/different Kaggle dataset
uv run python scripts/download_isprs_potsdam.py \
  --dataset-id aletbm/urban-segmentation-isprs
```

### Files Not Organized Correctly
If the script can't find images/labels automatically:

```bash
# Skip download and just organize
uv run python scripts/download_isprs_potsdam.py --skip-download
```

Or manually organize according to `data/isprs_potsdam/README.md`

---

## Alternative Datasets (Faster Download)

If the main dataset is too large or slow:

```bash
# Smaller Potsdam subset
uv run python scripts/download_isprs_potsdam.py \
  --dataset-id trito12/potsdam-vaihingen-isprs

# Urban segmentation (includes Potsdam)
uv run python scripts/download_isprs_potsdam.py \
  --dataset-id aletbm/urban-segmentation-isprs
```

---

## Script Options

```bash
# Full usage
uv run python scripts/download_isprs_potsdam.py --help

# Keep zip files after extraction
uv run python scripts/download_isprs_potsdam.py --keep-zips

# Custom output directory
uv run python scripts/download_isprs_potsdam.py \
  --output-dir data/my_custom_location

# Skip download (if files already exist)
uv run python scripts/download_isprs_potsdam.py --skip-download
```

---

## After Download

**Test the dataset:**
```bash
uv run python scripts/test_benchmark.py
```

**Run your first benchmark:**
```bash
uv run python scripts/run_benchmark.py \
  --dataset data/isprs_potsdam \
  --method v1.5 \
  --num-samples 5 \
  --save-viz
```

**Check results:**
```bash
ls -lh results/
```
