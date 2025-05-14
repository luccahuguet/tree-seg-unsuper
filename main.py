import sys
sys.path.append('src')
import kmeans_segmentation
import yaml

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    kmeans_segmentation.run_kmeans_segmentation(**config)
