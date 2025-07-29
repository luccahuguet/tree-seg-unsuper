---
layout: default
title: "Tree Segmentation Research"
nav_order: 1
---

<nav class="tree-seg-navbar">
  <div class="navbar-container">
    <a href="{{ '/' | relative_url }}" class="navbar-home">🌳 Tree Segmentation</a>
    <div class="navbar-links">
      <a href="{{ '/methodology' | relative_url }}">Methodology</a>
      <a href="{{ '/results' | relative_url }}">Results</a>
      <a href="{{ '/complete_example' | relative_url }}">Example</a>
      <a href="{{ '/parameter_comparison' | relative_url }}">Comparison</a>
      <a href="{{ '/analysis' | relative_url }}">Analysis</a>
    </div>
  </div>
</nav>

<style>
.tree-seg-navbar {
  background-color: #1e1e1e;
  border-bottom: 2px solid #00ff00;
  padding: 0.5rem 0;
  margin-bottom: 2rem;
}

.navbar-container {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 1rem;
}

.navbar-home {
  color: #00ff00 !important;
  text-decoration: none !important;
  font-weight: bold;
  font-size: 1.2rem;
}

.navbar-links {
  display: flex;
  gap: 1.5rem;
}

.navbar-links a {
  color: #fff !important;
  text-decoration: none !important;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: background-color 0.3s ease;
}

.navbar-links a:hover {
  background-color: #333;
  color: #00ff00 !important;
}

@media (max-width: 768px) {
  .navbar-container {
    flex-direction: column;
    gap: 1rem;
  }
  
  .navbar-links {
    flex-wrap: wrap;
    justify-content: center;
    gap: 1rem;
  }
}
</style>

# Tree Segmentation with DINOv2

## Overview

This research presents an unsupervised tree segmentation approach using DINOv2 Vision Transformers for aerial drone imagery. Our method combines modern deep learning features with intelligent clustering to achieve high-quality tree boundary detection.

## Key Features

- **🌳 Unsupervised Learning**: No manual annotations required
- **🚁 Drone Imagery**: Optimized for aerial perspectives
- **🤖 DINOv2 Features**: State-of-the-art vision transformer features
- **📊 Automatic K-Selection**: Intelligent cluster number detection
- **🎯 Professional Results**: Publication-ready visualizations

## Current Implementation

**Algorithm Version**: v1.5 (DINOv2 with patch + attention features)  
**Architecture**: Modern Python API with type safety  
**Clustering**: K-means with elbow method optimization  
**Visualization**: Multi-format outputs with config-based naming  

## Research Context

This work represents a modern approach to unsupervised segmentation in the forestry domain, leveraging recent advances in self-supervised learning to achieve robust tree boundary detection without requiring labeled training data.


