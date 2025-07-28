# GitHub Pages Setup

This folder contains the GitHub Pages site for presenting tree segmentation results.

## Quick Setup

1. **Generate results** with web optimization enabled:
   ```python
   from tree_seg import segment_trees
   results = segment_trees("image.jpg", web_optimize=True)
   ```
2. **Commit and push** changes
3. **Enable GitHub Pages** in repository settings

## Folder Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── index.md             # Homepage
├── methodology.md       # Technical approach
├── results.md           # Visual results
├── analysis.md          # Performance analysis
└── assets/images/       # Optimized images (auto-generated)
```

## Adding New Results

1. **Generate with web optimization**:
   ```python
   config = Config(web_optimize=True)
   segmenter = TreeSegmentation(config)
   results, paths = segmenter.process_and_visualize("image.jpg")
   ```
2. **Update results.md** with new experiment sections
3. **Commit and push**

## GitHub Pages Configuration

1. Go to repository **Settings** → **Pages**
2. Select **Deploy from a branch**
3. Choose **docs** folder from main branch
4. Site will be available at: `https://username.github.io/tree-seg-unsuper`

## Automatic Web Optimization

When `web_optimize=True` is enabled:
- **Compresses** 7MB images to ~1-2MB automatically
- **Converts** PNG to optimized JPEG
- **Resizes** to max 1200px width
- **Maintains** visual quality
- **No manual steps** required

## Professional Presentation

The site provides:
- **Clean academic layout** with navigation
- **Detailed methodology** section
- **Visual results** with descriptions
- **Performance analysis** and insights
- **Responsive design** for all devices

Perfect for sharing with professors and academic audiences!