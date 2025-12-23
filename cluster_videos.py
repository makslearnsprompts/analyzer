#!/usr/bin/env python3
"""
Incremental clustering of videos for A/B test detection.
"""

import argparse
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import everything we need from compare_two_videos.py
# (Ensure compare_two_videos.py is in the same directory)
try:
    from compare_two_videos import (
        init_gemini, 
        compare_videos, 
        get_videos_in_folder,
        BASE_DIR,
        DEFAULT_ANALYSIS_FPS,
        DEFAULT_FOCUSED_JUDGE_FPS
    )
except ImportError:
    print("❌ Error: compare_two_videos.py not found or not importable.")
    sys.exit(1)


class Cluster:
    def __init__(self, cluster_id: int, representative_video: Path):
        self.id = cluster_id
        self.representative = representative_video
        self.videos: List[Path] = [representative_video]
    
    def add_video(self, video: Path):
        self.videos.append(video)
        
    def to_dict(self):
        return {
            "id": self.id,
            "representative": self.representative.name,
            "videos": [v.name for v in self.videos],
            "count": len(self.videos)
        }


def prune_none(obj):
    """Recursively remove keys with None values from dicts/lists."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            out[k] = prune_none(v)
        return out
    if isinstance(obj, list):
        return [prune_none(v) for v in obj]
    return obj


def perform_clustering(folder_name: str, output_file: Optional[str] = None, max_workers: int = 4, fps: int = DEFAULT_ANALYSIS_FPS, judge_random_trimming: bool = True, judge_focused_trim: bool = True, judge_focused_fps: int = DEFAULT_FOCUSED_JUDGE_FPS, side_by_side: bool = True):
    folder = BASE_DIR / folder_name
    if not folder.exists():
        print(f"❌ Folder not found: {folder_name}")
        return

    videos = get_videos_in_folder(folder)
    if not videos:
        print(f"❌ No videos found in {folder_name}")
        return

    start_time = time.time()
    print(f"🚀 Starting incremental clustering for '{folder_name}' ({len(videos)} videos) (FPS={fps}, JudgeTrim={judge_random_trimming}, FocusTrim={judge_focused_trim}, SBS={side_by_side})")
    print("=" * 60)

    init_gemini()

    clusters: List[Cluster] = []
    comparisons_made = []
    remaining = videos.copy()
    cluster_id = 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while remaining:
            representative = remaining.pop(0)
            print(f"\n🎥 New cluster seed: {representative.name}")
            current_cluster = Cluster(cluster_id, representative)
            cluster_id += 1

            if not remaining:
                clusters.append(current_cluster)
                break

            candidates = remaining
            remaining = []

            futures = {executor.submit(compare_videos, representative, candidate, fps=fps, judge_random_trimming=judge_random_trimming, judge_focused_trim=judge_focused_trim, judge_focused_fps=judge_focused_fps, side_by_side=side_by_side): candidate for candidate in candidates}

            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logging.error("Comparison failed for %s vs %s: %s", representative.name, candidate.name, e)
                    result = {"same_group": None, "differences": [], "error": str(e)}

                comparisons_made.append({
                    "video_1": representative.name,
                    "video_2": candidate.name,
                    "same_group": result.get("same_group"),
                    "differences": result.get("differences", []),
                    "judge_logs": result.get("judge_logs"),
                    "error": result.get("error")
                })

                if result.get("same_group") is True:
                    print(f"   ✅ {candidate.name} matches cluster {current_cluster.id}")
                    current_cluster.add_video(candidate)
                else:
                    remaining.append(candidate)

            clusters.append(current_cluster)

    # --- Summary & Output ---
    print("\n" + "="*60)
    print("📊 CLUSTERING COMPLETE")
    print("="*60)
    
    for cluster in clusters:
        print(f"Cluster {cluster.id} ({len(cluster.videos)} videos):")
        for v in cluster.videos:
            print(f"  - {v.name}")
        print("-" * 20)

    # Save results
    output_data = {
        "folder": folder_name,
        "timestamp": datetime.now().isoformat(),
        "pipeline_seconds": round(time.time() - start_time, 3),
        "total_videos": len(videos),
        "total_clusters": len(clusters),
        "judge_random_trimming": judge_random_trimming,
        "judge_focused_trim": judge_focused_trim,
        "judge_focused_fps": judge_focused_fps,
        "judge_side_by_side": side_by_side,
        "clusters": [c.to_dict() for c in clusters],
        "comparisons_history": comparisons_made
    }
    output_data = prune_none(output_data)

    if not output_file:
        output_file = f"{folder_name}_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    logging.info("Clustering finished in %.2f seconds", time.time() - start_time)


def main():
    parser = argparse.ArgumentParser(description="Incremental Clustering for A/B Test Videos")
    parser.add_argument("--folder", "-f", required=True, help="Folder containing videos")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel comparisons worker count")
    parser.add_argument("--fps", type=int, default=DEFAULT_ANALYSIS_FPS, help="Frames per second for analysis")
    parser.add_argument("--no-judge-trim", action="store_false", dest="judge_trim", default=True, help="Disable random trimming for judge verification")
    parser.add_argument("--no-focused-trim", action="store_false", dest="focused_trim", default=True, help="Disable focused trimming for single difference")
    parser.add_argument("--focused-fps", type=int, default=DEFAULT_FOCUSED_JUDGE_FPS, help="FPS for focused judge analysis")
    parser.add_argument("--no-side-by-side", action="store_false", dest="side_by_side", default=True, help="Disable side-by-side video for focused judge")
    
    args = parser.parse_args()
    perform_clustering(args.folder, args.output, args.max_workers, fps=args.fps, judge_random_trimming=args.judge_trim, judge_focused_trim=args.focused_trim, judge_focused_fps=args.focused_fps, side_by_side=args.side_by_side)


if __name__ == "__main__":
    main()
