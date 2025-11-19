# Annotated Tree Dataset Search - Context Document

**Date:** 2025-11-18
**Project:** Unsupervised Tree Segmentation from Aerial Imagery
**Status:** V3 implementation phase (tree-specific logic)

---

## Project Background

We are developing an unsupervised tree segmentation pipeline using DINOv3 features + clustering for aerial/drone imagery. Current baseline (V1.5) achieves 22.5% mIoU on ISPRS Potsdam using general semantic segmentation metrics.

**Current architecture:**
- DINOv3 feature extraction (ViT-B/16 base model)
- K-means clustering with elbow method (auto K selection)
- SLIC boundary refinement
- Evaluation: Hungarian-matched mIoU against 6-class ground truth

**Next milestone (V3):** Add tree-specific logic:
- Vegetation filtering (ExG/CIVE indices)
- Cluster selection via IoU to vegetation mask
- Instance segmentation (distance transform + watershed)
- Shape/area filters with GSD awareness

---

## What We Need

An **annotated dataset of individual trees from aerial/drone/satellite top-down view** with:

### Minimum Requirements
- **Imagery type:** Aerial, drone, or satellite (top-down perspective)
- **Annotations:** Instance-level tree segmentation masks
  - Each tree as a separate instance/polygon
  - Binary tree/non-tree distinction
- **Format:** Common formats (COCO, Pascal VOC, GeoJSON, shapefiles, or labeled images)
- **Coverage:** Preferably diverse tree types and environments

### Desired Features (Priority Order)
1. **Species labels** per tree instance (enables V5 species purity evaluation)
2. **RGB + Multispectral** imagery (NIR, RedEdge for NDVI/GNDVI - V5 fusion)
3. **Similar to our use case:** Drone imagery at ~5-20cm GSD, temperate/tropical trees
4. **Moderate size:** 100+ annotated images or 1000+ tree instances
5. **Crown characteristics:** Tree height, crown diameter, health status (nice-to-have)

### What We DON'T Need
- ❌ Street-level/ground perspective images
- ❌ Single-tree close-ups (need landscape scenes with multiple trees)
- ❌ Semantic segmentation only (need instance separation)
- ❌ Lidar/point cloud only (need RGB imagery)

---

## Use Cases (Prioritized)

### 1. **Validation Dataset** (HIGH PRIORITY)
**Goal:** Quantitatively evaluate V3 tree detection performance
**Metrics needed:**
- Tree detection precision/recall (TP/FP/FN at instance level)
- IoU per tree instance
- F1 score for tree vs non-tree classification

**Minimum annotation:** Binary tree masks (tree/background) with instance separation

### 2. **Species Classification Ground Truth** (MEDIUM PRIORITY)
**Goal:** Enable V5 "species purity" metric - evaluate if clusters correspond to species
**Requirement:** Species labels per tree instance
**Metric:** Cluster homogeneity w.r.t. species labels

### 3. **Supervised Baseline Fine-tuning** (LOWER PRIORITY)
**Goal:** Fine-tune V4 Mask2Former on tree-specific data (currently zero-shot on ADE20K)
**Requirement:** Large dataset (1000+ trees) with consistent annotations
**Trade-off:** Time-intensive; only if unlocks significant gains

---

## Known Dataset Candidates

Please investigate and compare these (if they exist):

### Potential Sources
1. **TreeSatAI** - Satellite imagery with tree species classification
2. **NEON (National Ecological Observatory Network)** - Airborne imagery + lidar + field species data
3. **Individual Tree Crown (ITC) Datasets** - Various forestry research datasets
4. **ForestNet** - Global forest monitoring dataset
5. **DOTA/iSAID** - Aerial object detection (may include trees)
6. **Kaggle forestry competitions** - Tree crown detection/species challenges
7. **University research repositories** - Forestry/ecology departments often release datasets

### Geographic/Ecological Preferences
- **Temperate forests** preferred (our current focus)
- **Mixed species** (deciduous + coniferous)
- **Suburban/urban trees** also valuable (edges, mixed backgrounds)

---

## Search Strategy Recommendations

### Primary Keywords
- "individual tree crown dataset annotated"
- "tree species aerial imagery dataset"
- "tree segmentation instance masks drone"
- "forest canopy segmentation dataset top view"
- "tree crown delineation benchmark dataset"

### Secondary Keywords
- "urban tree inventory dataset aerial"
- "forestry remote sensing benchmark"
- "tree detection drone imagery labeled"
- "canopy species classification dataset"

### Platforms to Search
1. **Academic:** Google Dataset Search, Papers With Code datasets, IEEE DataPort
2. **Research repos:** Zenodo, Figshare, Dryad, university data repositories
3. **Competitions:** Kaggle, DrivenData, CodaLab
4. **Government/NGO:** USGS, ESA, NASA, forestry agencies

---

## Evaluation Criteria

For each dataset found, please assess:

### Critical Factors
- [ ] **Format compatibility:** Can we load it easily? (GeoJSON, COCO, labeled PNGs, etc.)
- [ ] **Annotation quality:** Are tree boundaries precise? Instance-level or semantic only?
- [ ] **Scale match:** Is the GSD/resolution similar to our use case (5-20cm)?
- [ ] **Licensing:** Can we use it for research/publication?

### Nice-to-Have Factors
- [ ] **Size:** How many annotated trees/images?
- [ ] **Species diversity:** Single species or multi-species?
- [ ] **Multispectral:** RGB-only or includes NIR/RedEdge?
- [ ] **Metadata:** GSD, location, acquisition date, camera specs?
- [ ] **Train/test split:** Pre-defined evaluation protocol?

---

## Dataset Format Preferences

### Ideal Formats (Best to Worst)
1. **COCO JSON** - Instance segmentation with polygons/RLE masks
2. **Labeled PNG pairs** - Image + instance mask (unique ID per tree)
3. **GeoJSON/Shapefiles** - Vector polygons (can rasterize to masks)
4. **Pascal VOC XML** - Bounding boxes (suboptimal, but usable)

### What to Avoid
- Proprietary formats requiring special software
- Lidar-only (need RGB imagery)
- Semantic masks without instance separation
- Extremely large files (>100GB) unless critical

---

## Time Constraints

**Context:** We're in a hurry to complete V3 implementation.

**Decision matrix:**
- **If excellent dataset found quickly:** Use it for V3 validation (high value)
- **If mediocre dataset found:** Assess if worth integration time
- **If no dataset found:** Proceed without; validate qualitatively only

**Time budget for dataset search:** 1-2 hours max
**Integration time budget:** 4-6 hours (format conversion, evaluation scripts)

---

## Deliverables from Search

Please provide:

1. **Top 3 dataset recommendations** with:
   - Name, source, download link
   - Brief description (size, annotations, format)
   - Pros/cons for our use case
   - Licensing info

2. **Quick comparison table:**
   - Dataset name | Instances | Species labels? | Multispectral? | Format | License

3. **Download/access instructions** for top choice

4. **Fallback plan** if no suitable dataset exists:
   - Could we use partial annotations (e.g., ADE20K tree class)?
   - Should we create minimal annotations from our own images?
   - Can we proceed with qualitative validation only?

---

## Questions to Answer

- Are there any **benchmark datasets** widely used for tree segmentation in remote sensing?
- What's the **current state-of-the-art** for individual tree crown delineation datasets?
- Do any datasets combine **RGB + multispectral + species labels**?
- Are there recent (2023-2025) **Kaggle competitions** on tree segmentation we could leverage?
- Any **active research groups** that might share data upon request?

---

## Contact Information (If Needed)

If datasets require data access requests or collaboration:
- We are academic researchers working on unsupervised forestry methods
- Can cite/acknowledge dataset providers in publications
- Open to collaboration if dataset owners interested in our methods

---

## Expected Output

**Preferred format for ChatGPT response:**

```markdown
## Dataset Recommendations

### Option 1: [Dataset Name]
- **Source:** [Link]
- **Size:** X images, Y tree instances
- **Annotations:** Instance masks / Species labels / etc.
- **Format:** COCO JSON / GeoJSON / etc.
- **Imagery:** RGB / Multispectral / Resolution
- **License:** [License type]
- **Pros:** ...
- **Cons:** ...
- **Download:** [Instructions]

### Option 2: [Dataset Name]
...

### Option 3: [Dataset Name]
...

## Comparison Table
| Dataset | Trees | Species? | Multi? | Format | License |
|---------|-------|----------|--------|--------|---------|
| ...

## Recommendation
[Which dataset to use and why, given our constraints]

## Fallback Strategy
[If no perfect match exists]
```

---

**Note:** Prioritize datasets that are:
1. **Immediately accessible** (no lengthy approval process)
2. **Well-documented** (clear format, example code)
3. **Similar to our imagery** (drone/aerial, moderate GSD)
4. **Instance-level annotations** (not just semantic masks)

Good luck with the search!
