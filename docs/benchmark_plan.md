# Benchmark Evaluation Plan: ISPRS Potsdam/Vaihingen

## Goal
Establish quantitative baseline for comparing V1/V2/V3 architectures using standard aerial imagery segmentation benchmark.

---

## Phase 1: Dataset Preparation

### 1.1 Download ISPRS Potsdam Dataset
- **Source**: https://www.isprs.org/education/benchmarks/UrbanSemLab/
- **What to download**:
  - RGB orthophotos (Top_Potsdam_*.tif)
  - Ground truth labels (Top_Potsdam_*_label.tif)
  - ~38 high-resolution aerial images (6000x6000 pixels)
- **Classes** (6 total):
  - 0: Impervious surfaces (roads, parking)
  - 1: Building
  - 2: Low vegetation (grass, small plants)
  - 3: Tree
  - 4: Car
  - 5: Clutter/background

### 1.2 Dataset Storage Structure
```
data/
├── isprs_potsdam/
│   ├── images/          # RGB orthophotos
│   │   ├── top_potsdam_2_10.tif
│   │   ├── top_potsdam_2_11.tif
│   │   └── ...
│   ├── labels/          # Ground truth semantic masks
│   │   ├── top_potsdam_2_10_label.tif
│   │   ├── top_potsdam_2_11_label.tif
│   │   └── ...
│   └── README.md        # Dataset documentation
```

### 1.3 Preprocessing Requirements
- Convert TIFF to PNG/JPG if needed
- Handle large images (6000x6000):
  - Option A: Tile into 1024x1024 patches
  - Option B: Downsample to manageable size (2048x2048)
  - Option C: Process full-res (memory intensive)
- Verify label format (integer-encoded classes 0-5)

---

## Phase 2: Evaluation Infrastructure

### 2.1 Create Evaluation Module
**File**: `tree_seg/evaluation/metrics.py`

```python
# Core functionality needed:
- compute_miou(pred_labels, gt_labels, num_classes)
- compute_pixel_accuracy(pred_labels, gt_labels)
- hungarian_matching(pred_clusters, gt_labels)
  # Match K predicted clusters to ground truth classes
```

### 2.2 Create Benchmark Runner
**File**: `tree_seg/evaluation/benchmark.py`

```python
# Workflow:
1. Load dataset (images + labels)
2. Run segmentation method (V1/V2/V3)
3. Match predicted clusters to GT classes (Hungarian algorithm)
4. Compute metrics (mIoU, pixel accuracy)
5. Generate results table
6. Save visualizations (predictions vs GT)
```

### 2.3 Add Configuration Support
**Update**: `tree_seg/core/types.py`

```python
# Add optional fields to Config:
- eval_mode: bool = False
- ground_truth_path: Optional[Path] = None
- eval_metrics: List[str] = ["miou", "pixel_accuracy"]
```

---

## Phase 3: Implementation Tasks

### 3.1 Hungarian Algorithm Matching
**Challenge**: Predicted clusters (0-K) need to map to GT classes (0-5)
**Solution**: Hungarian algorithm finds optimal cluster→class assignment

**Implementation**:
- Use `scipy.optimize.linear_sum_assignment`
- Build confusion matrix between predictions and GT
- Find assignment that maximizes agreement

### 3.2 Metric Computation
**mIoU (mean Intersection over Union)**:
```
For each class c:
  IoU_c = (pred ∩ gt) / (pred ∪ gt)
mIoU = mean(IoU_c) across all classes
```

**Pixel Accuracy**:
```
accuracy = (correct_pixels) / (total_pixels)
```

### 3.3 Batch Processing Script
**File**: `scripts/run_benchmark.py`

```bash
# Usage:
python scripts/run_benchmark.py \
  --dataset isprs_potsdam \
  --method v1 \
  --model base \
  --output results/v1_base_potsdam.json
```

---

## Phase 4: Running Benchmarks

### 4.1 Baseline Evaluation (V1.5)
```bash
# Test with different configurations:
- V1.5 + base model + elbow_threshold=5.0
- V1.5 + base model + elbow_threshold=10.0
- V1.5 + large model + elbow_threshold=5.0
- V1.5 + mega model + elbow_threshold=5.0
```

### 4.2 Expected Results Table
```markdown
| Method | Model | Elbow Threshold | K (avg) | mIoU | Pixel Acc |
|--------|-------|-----------------|---------|------|-----------|
| V1.5   | base  | 5.0             | ?       | ?    | ?         |
| V1.5   | base  | 10.0            | ?       | ?    | ?         |
| V1.5   | large | 5.0             | ?       | ?    | ?         |
| V1.5   | mega  | 5.0             | ?       | ?    | ?         |
```

### 4.3 Visualization Outputs
For each test image, generate:
- Original image
- Predicted segmentation (colored clusters)
- Ground truth segmentation
- Confusion matrix heatmap

---

## Phase 5: V2/V3 Comparison (Future)

### 5.1 When U2Seg (V2) is implemented
- Run same benchmark with V2
- Compare: V1.5 vs V2 on same images
- Analyze: speed, memory, quality trade-offs

### 5.2 When DynaSeg (V3) is implemented
- Run same benchmark with V3
- Full comparison: V1.5 vs V2 vs V3

### 5.3 Comparison Metrics
- mIoU (primary)
- Pixel accuracy (secondary)
- Runtime (seconds per image)
- Memory usage (GB)
- Cluster stability (multiple runs)

---

## Timeline (Week 1 Goal)

### Day 1-2: Dataset Setup
- [ ] Download ISPRS Potsdam dataset
- [ ] Organize into `data/isprs_potsdam/` structure
- [ ] Verify image/label pairs load correctly
- [ ] Decide on tiling/downsampling strategy

### Day 3-4: Evaluation Code
- [ ] Implement `metrics.py` (mIoU, pixel accuracy, Hungarian matching)
- [ ] Implement `benchmark.py` (batch runner)
- [ ] Add eval mode to Config
- [ ] Test on 1-2 sample images

### Day 5: Run V1.5 Baseline
- [ ] Run V1.5 on full Potsdam test set
- [ ] Generate results table
- [ ] Create visualization comparisons
- [ ] Document findings

### Day 6-7: Analysis & Write-up
- [ ] Analyze V1.5 performance across different configs
- [ ] Identify strengths/weaknesses
- [ ] Write methodology section draft
- [ ] Prepare for V2/V3 comparison

---

## Success Criteria

✅ **Week 1 Complete When**:
1. ISPRS Potsdam dataset downloaded and processed
2. Evaluation metrics implemented and tested
3. V1.5 baseline results on full test set
4. Results table with mIoU and pixel accuracy
5. Visualizations showing predictions vs ground truth
6. Infrastructure ready for V2/V3 comparison

---

## Alternative: Quick Start with Subset

If time is tight, start with:
- 5 images from Potsdam (not full 38)
- Qualitative comparison first, metrics second
- Iterate quickly to validate approach

---

## Notes

- **ISPRS has train/test split**: Use test split for evaluation (no GT labels public for test, use "train" for your validation)
- **Cluster count**: ISPRS has 6 classes, but K may be 5-15 (more granular clustering is OK)
- **Tree class**: Class 3 in ISPRS labels = trees (useful for future tree-specific work)
- **Computational cost**: 6000x6000 images are large; tiling recommended for speed

---

## Future Extensions

1. **Add more datasets**: LoveDA, Agriculture-Vision
2. **Tree-specific evaluation**: Filter to only evaluate "tree" class vs others
3. **Boundary metrics**: Boundary F1-score, edge precision/recall
4. **Cluster interpretability**: PCA visualization of matched clusters
5. **Cross-dataset generalization**: Train on Potsdam, test on Vaihingen
