#!/usr/bin/env python3
"""
Elegant benchmark viewer for DINOv3 performance data.
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime


def load_benchmarks(file_path="data/outputs/performance_log.jsonl"):
    """Load benchmark data from JSONL file."""
    log_file = Path(file_path)
    if not log_file.exists():
        print(f"❌ Performance log not found: {log_file}")
        return None

    benchmarks = []
    with open(log_file) as f:
        content = f.read().strip()
        # Handle pretty-printed JSON by splitting on }{ patterns
        if content.startswith("{\n"):  # Pretty printed
            entries = content.split("}\n{")
            for i, entry in enumerate(entries):
                if i > 0:
                    entry = "{" + entry
                if i < len(entries) - 1:
                    entry = entry + "}"
                if entry.strip():
                    benchmarks.append(json.loads(entry))
        else:  # Regular JSONL
            for line in f:
                if line.strip():
                    benchmarks.append(json.loads(line))

    return pd.DataFrame(benchmarks)


def format_model_name(name):
    """Format model name for display."""
    mapping = {
        "dinov3_vits16": "Small (S)",
        "dinov3_vitb16": "Base (B)",
        "dinov3_vitl16": "Large (L)",
        "dinov3_vith16plus": "Huge+ (H+)",
        "dinov3_vit7b16": "Mega (7B)",
    }
    return mapping.get(name, name)


def view_summary(df):
    """Display performance summary table."""
    if df is None or df.empty:
        print("❌ No benchmark data available")
        return

    print("🚀 DINOv3 Performance Summary")
    print("=" * 80)

    # Sort by model complexity
    model_order = [
        "dinov3_vits16",
        "dinov3_vitb16",
        "dinov3_vitl16",
        "dinov3_vith16plus",
        "dinov3_vit7b16",
    ]
    df["sort_order"] = df["model_name"].map(
        lambda x: model_order.index(x) if x in model_order else 999
    )
    df = df.sort_values("sort_order")

    # Format display table
    display_df = df.copy()
    display_df["Model"] = display_df["model_name"].map(format_model_name)
    display_df["Parameters"] = display_df["model_params"]
    display_df["Features"] = display_df["feature_dims"].astype(str) + "D"
    display_df["Total (s)"] = display_df["total_time_s"].round(3)
    display_df["K-Selection (s)"] = display_df["k_selection_s"].round(3)
    display_df["Clustering (s)"] = display_df["clustering_s"].round(3)
    display_df["K Value"] = display_df["optimal_k"]
    display_df["GPU Memory"] = display_df["gpu_used"].map({True: "✅", False: "❌"})

    # Select columns for display
    cols = [
        "Model",
        "Parameters",
        "Features",
        "Total (s)",
        "K-Selection (s)",
        "Clustering (s)",
        "K Value",
        "GPU Memory",
    ]

    print("\n📊 Performance Comparison")
    print("-" * 80)
    print(display_df[cols].to_string(index=False, max_colwidth=12))

    return display_df


def view_analysis(df):
    """Display detailed performance analysis."""
    if df is None or df.empty:
        return

    print("\n📈 Performance Analysis")
    print("-" * 50)

    # Convert parameters to numeric for analysis
    def parse_params(param_str):
        if "B" in param_str:
            return float(param_str.replace("B", "")) * 1000
        else:
            return float(param_str.replace("M", ""))

    df["params_numeric"] = df["model_params"].map(parse_params)
    base_time = df["total_time_s"].min()
    base_params = df["params_numeric"].min()

    df["time_ratio"] = df["total_time_s"] / base_time
    df["param_ratio"] = df["params_numeric"] / base_params
    df["efficiency"] = df["param_ratio"] / df["time_ratio"]

    # Scaling analysis
    for _, row in df.iterrows():
        model = format_model_name(row["model_name"])
        print(
            f"{model:<12} {row['time_ratio']:.1f}x time  {row['param_ratio']:.1f}x params  Efficiency: {row['efficiency']:.1f}"
        )

    print("\n🎯 Key Insights")
    print("-" * 30)

    # K-selection percentage
    k_pct = (df["k_selection_s"].mean() / df["total_time_s"].mean()) * 100
    print(f"• K-selection: {k_pct:.1f}% of total time")
    print(f"• Optimal K range: {df['optimal_k'].min()}-{df['optimal_k'].max()}")
    print(
        f"• Feature dimensions: {df['feature_dims'].min()}D → {df['feature_dims'].max()}D"
    )
    print(
        f"• Parameter range: {df['params_numeric'].min():.0f}M → {df['params_numeric'].max():.0f}M"
    )
    print(
        f"• Speed range: {df['total_time_s'].min():.2f}s → {df['total_time_s'].max():.2f}s"
    )


def view_raw(df):
    """Display raw JSON data in readable format."""
    if df is None or df.empty:
        return

    print("\n📋 Raw Benchmark Data")
    print("-" * 50)

    for i, row in df.iterrows():
        print(f"\n[{i + 1}] {format_model_name(row['model_name'])}")
        timestamp = datetime.fromisoformat(row["timestamp"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"    Timestamp: {timestamp}")
        print(
            f"    Image: {Path(row['image_path']).name} ({row['image_size'][0]}x{row['image_size'][1]})"
        )
        print(f"    Total Time: {row['total_time_s']:.3f}s")
        print("    Breakdown:")
        print(
            f"      - K-Selection: {row['k_selection_s']:.3f}s ({row['k_selection_s'] / row['total_time_s'] * 100:.1f}%)"
        )
        print(
            f"      - Clustering: {row['clustering_s']:.3f}s ({row['clustering_s'] / row['total_time_s'] * 100:.1f}%)"
        )
        print(
            f"    Results: K={row['optimal_k']}, Features={row['feature_dims']}D, WCSS={row['final_wcss']:.0f}"
        )


def main():
    parser = argparse.ArgumentParser(description="View DINOv3 benchmark results")
    parser.add_argument(
        "--file",
        "-f",
        default="data/outputs/performance_log.jsonl",
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["summary", "analysis", "raw", "all"],
        default="all",
        help="Display mode",
    )
    parser.add_argument("--latest", "-l", type=int, help="Show only latest N entries")

    args = parser.parse_args()

    # Load data
    df = load_benchmarks(args.file)
    if df is None:
        return

    # Filter to latest entries if requested
    if args.latest:
        df = df.tail(args.latest)

    # Display based on mode
    if args.mode in ["summary", "all"]:
        view_summary(df)

    if args.mode in ["analysis", "all"]:
        view_analysis(df)

    if args.mode in ["raw", "all"]:
        view_raw(df)

    print(f"\n📁 Data from: {args.file}")
    print(f"📊 Total entries: {len(df)}")


if __name__ == "__main__":
    main()
