# Tree Species Segmentation Dataset Comparison

Comprehensive comparison of available datasets for species-level semantic segmentation in aerial imagery.

**Last Updated:** 2025-01-26
**Context:** Evaluating datasets for DINOv3-based species clustering research with 2.5 week timeline

---

## Executive Summary

**Top Recommendation:** FORTRESS (Black Forest, Germany)
**Best Alternative:** Quebec Trees (Canada)
**Generalization Test:** Savanna Trees (Australia)

**Avoid:** TreeSatAI, PureForest2024 (patch classification, not segmentation)

---

## Quick Comparison Table

| Dataset | Type | Species | Size | GSD | Area | License | Download |
|---------|------|---------|------|-----|------|---------|----------|
| **FORTRESS** | Semantic seg | 9 spp + 3 genera | 40.5 GB | <2 cm | 51 ha | CC BY 4.0 | [RADAR@KIT](https://radar.kit.edu/radar/en/dataset/AwwREVscwqlcTVSw) |
| **Quebec Trees** | Instance seg | 14 species | 150 GB (20 GB/date) | ~5 cm | 22k trees | CC BY 4.0 | [Zenodo](https://zenodo.org/records/8148479) |
| **Savanna Trees** | Instance seg | 36 species | 8 GB | ~2 cm | 2.5k trees | CC BY 4.0 | [Zenodo](https://zenodo.org/records/7094916) |
| **OAM-TCD** | Instance seg | No species | 80 GB | 10 cm | 280k trees | CC BY 4.0 | [Zenodo](https://zenodo.org/records/12516583) |
| TreeSatAI ❌ | Classification | 20 classes | ~50 GB | 20 cm | 50k patches | CC BY 4.0 | [Zenodo](https://zenodo.org/records/6598390) |
| PureForest2024 ❌ | Classification | 18 species | 339 km² | 20 cm | 135k patches | Open 2.0 | [HuggingFace](https://huggingface.co/datasets/IGNF/PureForest) |

---

## Dataset Details

### 1. FORTRESS (Forest Tree Species Segmentation)

**Source:** Karlsruhe Institute of Technology (Schiefer, Frey, Kattenborn et al., 2022)
**DOI:** 10.35097/538
**Paper:** [Schiefer et al. 2020, ISPRS J. Photogramm.](https://doi.org/10.1016/j.isprsjprs.2020.10.015)

#### Overview
UAV RGB orthomosaics with pixel-level semantic segmentation masks for temperate forest species in Germany.

#### Specifications
- **Coverage:** 51 hectares (Southern Black Forest + Hainich National Park)
- **Species:** 9 tree species + 3 genus-level classes + deadwood + forest floor (~13-14 total classes)
  - Likely includes: *Fagus* (beech), *Picea* (spruce), *Abies* (fir), and other Central European temperate species
- **Segmentation:** Semantic (contiguous same-species regions labeled as patches)
- **Resolution:** <2 cm GSD (very high resolution)
- **Acquisition:** 2017-2019, leaf-on conditions
- **Format:**
  - Georeferenced RGB orthomosaics (GeoTIFF)
  - Pixel-wise species masks (multiclass raster)
- **Size:** 40.5 GB total
- **Baseline Performance:** F1-score 0.73 (U-Net)

#### ✅ Pros
- **Perfect segmentation type:** True semantic segmentation with species patches already delineated
- **Highest resolution:** <2 cm GSD provides exceptional detail for DINOv3 feature extraction
- **Ready-to-use format:** Orthomosaics + masks, no preprocessing required
- **Manageable size:** 40.5 GB total (single download)
- **Established benchmark:** Published baseline (F1=0.73) for comparison
- **Clean labels:** Expert-derived species annotations
- **Mixed forest conditions:** Natural temperate forest with coniferous + deciduous mix
- **Open license:** CC BY 4.0, free for research

#### ❌ Cons
- **Limited geographic diversity:** Single region (Germany), limited to Central European species
- **Moderate species count:** 9 species (fewer than Quebec's 14 or Savanna's 36)
- **RGB only:** No NIR or multispectral bands
- **Unknown species distribution:** Species balance/class distribution not publicly documented
- **No individual trees:** Semantic patches don't distinguish individual tree crowns (but this matches your use case)
- **Limited documentation:** Dataset has metadata but no detailed benchmark paper beyond the archive
- **Smaller scale:** 51 ha is moderate compared to larger multi-site datasets

#### Best For
- **Species-level semantic segmentation** (patches, not individual trees)
- **High-resolution feature extraction** (DINOv3 excels at <5 cm GSD)
- **Temperate forest ecosystems**
- **Quick integration** (2-3 days)

#### Integration Complexity
**Low** (2-3 days)
- Day 1: Download + verify format
- Day 2: Write data loader, inspect classes
- Day 3: Test pipeline

---

### 2. Quebec Trees Dataset (Multi-Temporal UAV)

**Source:** University of Montréal & Univ. Sherbrooke (Cloutier, Germain, Laliberté, 2023)
**DOI:** 10.5281/zenodo.8148479
**Paper:** [Cloutier et al. 2023, bioRxiv](https://doi.org/10.1101/2023.08.03.548604)

#### Overview
Multi-season UAV RGB orthomosaics with individual tree crown polygons labeled by species.

#### Specifications
- **Coverage:** ~22,000 labeled trees over one temperate-mixed forest site (Quebec, Canada)
- **Species:** 14 species classes (mostly species-level, some genus-level)
  - Examples: balsam fir, red maple, sugar maple, yellow birch, paper birch, spruce spp., aspen spp., eastern hemlock, etc.
- **Segmentation:** Instance segmentation (individual tree crown polygons)
- **Resolution:** ~5 cm GSD
- **Acquisition:** 7 dates (May-October 2021) - captures phenological variation
- **Format:**
  - RGB orthomosaics (Cloud-Optimized GeoTIFF)
  - Tree crown polygons with species labels (GeoPackage .gpkg)
  - Photogrammetric point clouds (COPC .laz) - optional
- **Size:** ~150 GB total (7 temporal snapshots, ~20-23 GB each)
- **Study area:** 3 zones covering the same forest site

#### ✅ Pros
- **Large sample size:** 22,000+ individual trees (excellent for validation)
- **High species diversity:** 14 species covering mixed temperate forest
- **Multi-temporal:** 7 acquisition dates capture seasonal phenology (leaf-on/off)
- **Expert-verified labels:** Manually delineated and tagged by botanists
- **Precise annotations:** Vector polygons for each tree crown with species codes
- **Standard GIS formats:** COG + GeoPackage, easy integration
- **Well-documented:** Active research use, clear metadata, recent publication
- **Natural mixed forest:** Realistic complex scenario (maples, birches, conifers, etc.)
- **Open license:** CC BY 4.0

#### ❌ Cons
- **Very large download:** 150 GB total (or 20+ GB per temporal snapshot if selecting one date)
- **Single geographic location:** One forest site in Quebec, no landscape diversity
- **RGB only:** No NIR or multispectral bands
- **Instance → Semantic conversion required:** Polygons need rasterization and merging of touching same-species trees
- **Some genus-level labels:** 1,956 trees labeled only at genus (species ID too difficult)
- **Temporal redundancy:** 7 dates provide phenology but may be overkill for your use case
- **Similar species challenges:** Some visually similar species merged at genus level

#### Best For
- **Large-scale validation:** 22k trees provide robust statistics
- **Species diversity:** 14 species > FORTRESS's 9
- **Phenological analysis:** Multi-temporal if you want to test seasonal robustness
- **North American temperate forests**

#### Integration Complexity
**Medium** (3-4 days)
- Day 1: Download one temporal snapshot (~20 GB)
- Day 2: Write GeoPackage reader, rasterize polygons
- Day 3: Merge touching same-species trees → semantic masks
- Day 4: Test pipeline

---

### 3. Northern Australia Savanna Trees Dataset

**Source:** James Cook University (Jansen et al., 2023)
**DOI:** 10.5281/zenodo.7094916
**Paper:** [Jansen et al. 2023, MDPI Data](https://doi.org/10.3390/data8020044)

#### Overview
Drone RGB imagery with individual tree crown polygons for tropical savanna species.

#### Specifications
- **Coverage:** 2,547 tree instances across 7 hectares (1 ha plots)
- **Species:** 36 tree species (tropical savanna taxa)
  - Examples: *Eucalyptus tetrodonta*, *E. miniata*, *Acacia* spp., etc.
- **Segmentation:** Instance segmentation (individual tree crowns)
- **Resolution:** ~1.8-2.0 cm GSD (ultra high resolution)
- **Acquisition:** DJI drone at ~80m altitude, Northern Australia (Kakadu National Park)
- **Format:**
  - Georeferenced RGB orthomosaics (GeoTIFF) - 7 images, ~1 ha each
  - Vector crown polygons (Shapefile/GeoJSON) with species attributes
  - COCO-format tiles (1024×1024 px) with JSON annotations (ML-ready)
- **Size:** ~8 GB total (COCO tiles + original orthos)
- **Environment:** Open woodland (scattered trees over grassland)

#### ✅ Pros
- **Highest species diversity:** 36 species (taxonomically rich)
- **Ultra high resolution:** ~2 cm GSD (best spatial detail)
- **Ground-truthed labels:** Field surveys + botanist verification (very accurate)
- **ML-ready format:** COCO JSON for instance segmentation (plug-and-play)
- **Small dataset size:** 8 GB, quick to download and test
- **Dual format:** GIS (Shapefile) + ML (COCO) versions provided
- **Clear documentation:** Data journal article with baseline results
- **Simple background:** Grassland makes crowns well-defined (less occlusion)
- **Unique ecosystem:** Complements temperate forest datasets (generalization test)

#### ❌ Cons
- **Small sample size:** Only 2,547 trees (low for deep learning training)
- **Imbalanced classes:** Some dominant species have hundreds, many rare species have <10 samples
- **Specialized use case:** Tropical savanna (dry Australia) - models may not generalize to dense forests
- **Limited coverage:** Only 7 hectares across 7 plots
- **RGB only:** No multispectral (some species subtle without NIR)
- **Different ecosystem:** Open woodland ≠ dense forest (crown overlap minimal)
- **Geographic specificity:** Australian species not found elsewhere

#### Best For
- **Generalization testing:** Cross-ecosystem validation (tropical vs temperate)
- **High taxonomic diversity:** 36 species tests multi-class discrimination
- **Quick experiments:** Small size (8 GB) allows rapid iteration
- **Instance segmentation baselines:** COCO format ready for Mask R-CNN, etc.

#### Integration Complexity
**Low** (1-2 days)
- Day 1: Download + load COCO format or rasterize shapefiles
- Day 2: Test pipeline

---

### 4. OAM-TCD (OpenAerialMap Tree Cover Dataset)

**Source:** Howe et al., 2024
**DOI:** 10.5281/zenodo.12516583
**Paper:** [Howe et al. 2024, arXiv](https://arxiv.org/abs/2407.11743)

#### Overview
Your current dataset - globally diverse high-resolution tree cover maps with instance masks (no species labels).

#### Specifications
- **Coverage:** 280,000+ individual trees + 56,000 groups across global locations
- **Species:** **None** (only "tree" vs "group of trees")
- **Segmentation:** Instance segmentation (MS-COCO format)
- **Resolution:** 10 cm GSD
- **Acquisition:** OpenAerialMap imagery (various sources)
- **Format:**
  - 5,072 images (2048×2048 px)
  - Instance masks (polygons)
  - MS-COCO JSON annotations
- **Size:** ~80 GB

#### ✅ Pros
- **Already integrated:** You've done V3 evaluation on this
- **Large scale:** 280k trees, globally diverse
- **High resolution:** 10 cm GSD (good for DINOv3)
- **Instance masks:** Can merge to semantic regions
- **Global diversity:** Various ecosystems, locations

#### ❌ Cons
- **NO SPECIES LABELS:** Cannot validate species clustering claims
- **Groups vs individuals:** "Group of trees" labels are semantic but not species-specific
- **Qualitative only:** Results cannot be quantitatively validated for species accuracy
- **Limited scientific contribution:** Without species ground truth, paper lacks rigor

#### Best For
- **Vegetation vs non-vegetation:** V3 vegetation filtering validation
- **Not suitable for:** Species-level clustering validation

#### Integration Complexity
**Already done** (0 days)

---

### 5. TreeSatAI ❌ (Not Recommended)

**Source:** Ahlswede et al., 2023
**DOI:** 10.5281/zenodo.6598390
**Paper:** [ESSD 2023](https://essd.copernicus.org/articles/15/681/2023/)

#### Overview
Multi-sensor forest species classification dataset (patch-level, NOT segmentation).

#### Specifications
- **Coverage:** 50,381 patches (60m×60m) in Lower Saxony, Germany
- **Species:** 20 tree species (15 genera)
- **Segmentation:** **NONE** - Patch-level classification only
- **Resolution:** 20 cm aerial RGB + Sentinel-1/2
- **Format:** Image patches with single dominant species label per patch
- **Size:** ~50 GB

#### ❌ Why Not Suitable
- **No segmentation masks:** Each 60m patch has ONE label (dominant species)
- **No pixel-level labels:** Cannot identify where species boundaries are within patches
- **Classification task:** Not designed for semantic/instance segmentation
- **Wrong data format:** Doesn't match your research needs

#### Could Be Used For
- Patch-level classification (not your use case)
- Multi-sensor fusion experiments (if you had Sentinel data)

---

### 6. PureForest2024 ❌ (Not Recommended)

**Source:** IGNF (Institut National de l'Information Géographique et Forestière), 2024
**Paper:** [arXiv 2404.12064](https://arxiv.org/abs/2404.12064)

#### Overview
Large-scale aerial LiDAR + imagery dataset for monospecific forest classification.

#### Specifications
- **Coverage:** 135,569 patches (50m×50m), 339 km² across southern France
- **Species:** 18 tree species grouped into 13 semantic classes
- **Segmentation:** **NONE** - Patch classification (monospecific forests)
- **Resolution:** 20 cm aerial imagery + LiDAR (~40 pts/m²)
- **Format:**
  - 250×250 px image tiles (GeoTIFF)
  - LiDAR point clouds (LAZ)
  - Single label per 50m patch
- **Size:** Very large (339 km²)

#### ❌ Why Not Suitable
- **Monospecific forests:** Each patch contains ONE species (pure stands)
- **No segmentation masks:** Patch classification, not pixel-level segmentation
- **Wrong spatial scale:** 50m patches too coarse for species boundaries
- **Classification task:** Not designed for semantic segmentation

#### Could Be Used For
- Forest stand classification (not your use case)
- LiDAR + imagery fusion (if you had LiDAR pipeline)

---

## Dataset Selection Guide

### For Your Use Case: Species-Level Semantic Segmentation

**Timeline:** 2.5 weeks (testing + paper writing)

#### Ranking

1. **FORTRESS** ⭐⭐⭐⭐⭐
   - **Perfect match:** Semantic segmentation of species patches
   - **Lowest risk:** Ready-to-use format, manageable size
   - **Timeline:** 2-3 days integration

2. **Quebec Trees** ⭐⭐⭐⭐
   - **Excellent alternative:** More species, larger sample
   - **Higher effort:** Instance→semantic conversion required
   - **Timeline:** 3-4 days integration

3. **Savanna Trees** ⭐⭐⭐
   - **Bonus dataset:** Generalization testing (cross-ecosystem)
   - **Quick addition:** 1-2 days after primary dataset
   - **Timeline:** Adds minimal time, strengthens paper

4. **OAM-TCD** ⭐⭐
   - **Current baseline:** Good for vegetation filtering validation
   - **Not sufficient:** Lacks species labels for main claims

#### Avoid Entirely
- ❌ **TreeSatAI:** Patch classification, not segmentation
- ❌ **PureForest2024:** Monospecific patch classification, not segmentation

---

## Recommended Strategy (2.5 Weeks)

### Week 1: FORTRESS Primary Evaluation
- **Days 1-3:** Download FORTRESS, integrate, run V1.5 + V3
- **Days 4-6:** Compute species-level metrics, generate figures
- **Day 7:** Results analysis

### Week 2: Generalization + Writing
- **Day 8:** Download Savanna Trees (8 GB)
- **Day 9:** Test V3 generalization on tropical ecosystem
- **Days 10-14:** Write paper (methods, results, discussion)

### Week 2.5: Polish
- **Days 15-17:** Abstract, intro, revisions, submission

### Backup Plan
If FORTRESS has format issues (discovered Day 1-2):
- Switch to **Quebec Trees** (download single temporal snapshot)
- Adjust timeline: 4 days integration instead of 3
- Still achievable within 2.5 weeks

---

## Key Metrics You Can Compute

### With FORTRESS or Quebec (species labels)
- **Per-species IoU:** Maple vs Birch vs Fir clustering quality
- **Species purity:** % of each DINOv3 cluster that's a single species
- **Boundary accuracy:** Edge-F score for species boundaries
- **Confusion matrix:** Which species V3 confuses (e.g., spruce vs fir)
- **Cluster-to-species mapping:** Hungarian algorithm assignment quality
- **Species recall/precision:** Does V3 successfully separate each species?

### With OAM-TCD only (no species)
- ❌ Cannot validate species clustering
- ✅ Can validate vegetation vs non-vegetation filtering

---

## Download Commands

### FORTRESS
```bash
# Visit RADAR@KIT and download via browser (institutional access may help)
wget https://radar.kit.edu/radar/en/dataset/AwwREVscwqlcTVSw.FORTRESS
# Or use institutional credentials if available
```

### Quebec Trees (Single Snapshot)
```bash
# Download July 2021 snapshot (~20 GB)
wget https://zenodo.org/records/8148479/files/quebec_trees_dataset_2021-07-21.zip
```

### Savanna Trees
```bash
# Download full COCO dataset (~8 GB)
wget https://zenodo.org/records/7094916/files/savanna_trees_coco.zip
```

---

## Citations

### FORTRESS
```bibtex
@dataset{schiefer_2022_fortress,
  author       = {Schiefer, Felix and Frey, Julian and Kattenborn, Teja},
  title        = {FORTRESS: Forest Tree Species Segmentation},
  year         = {2022},
  publisher    = {Karlsruhe Institute of Technology},
  doi          = {10.35097/538},
  url          = {https://radar.kit.edu/radar/en/dataset/AwwREVscwqlcTVSw}
}

@article{schiefer_2020_isprs,
  title={Mapping forest tree species in high resolution UAV-based RGB-imagery by means of convolutional neural networks},
  author={Schiefer, Felix and Kattenborn, Teja and Frick, Alice and Frey, Julian and Schall, Peter and Koch, Barbara and Schmidtlein, Sebastian},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={170},
  pages={205--215},
  year={2020},
  doi={10.1016/j.isprsjprs.2020.10.015}
}
```

### Quebec Trees
```bibtex
@dataset{cloutier_2023_quebec,
  author       = {Cloutier, Mikaël and Germain, Maxence and Laliberté, Etienne},
  title        = {Quebec Trees Dataset},
  year         = {2023},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.8148479},
  url          = {https://zenodo.org/records/8148479}
}
```

### Savanna Trees
```bibtex
@article{jansen_2023_savanna,
  title={Deep Learning with Northern Australian Savanna Tree Species: A Novel Dataset},
  author={Jansen, A. J. and others},
  journal={Data},
  volume={8},
  number={2},
  pages={44},
  year={2023},
  doi={10.3390/data8020044}
}
```

---

## Resources

- [FORTRESS on RADAR@KIT](https://radar.kit.edu/radar/en/dataset/AwwREVscwqlcTVSw)
- [Quebec Trees on Zenodo](https://zenodo.org/records/8148479)
- [Savanna Trees on Zenodo](https://zenodo.org/records/7094916)
- [OAM-TCD on Zenodo](https://zenodo.org/records/12516583)
- [awesome-forests GitHub](https://github.com/blutjens/awesome-forests)
- [OpenForest Catalog](https://github.com/RolnickLab/OpenForest)

---

**Last Updated:** 2025-01-26
**Next Steps:** Download FORTRESS and begin integration
