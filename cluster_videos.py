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

# Import everything we need from compare_two_videos.py
# (Ensure compare_two_videos.py is in the same directory)
try:
    from compare_two_videos import (
        init_gemini, 
        compare_videos, 
        load_prompt_v2, 
        get_videos_in_folder,
        BASE_DIR
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


def perform_clustering(folder_name: str, output_file: Optional[str] = None):
    folder = BASE_DIR / folder_name
    if not folder.exists():
        print(f"❌ Folder not found: {folder_name}")
        return

    videos = get_videos_in_folder(folder)
    if not videos:
        print(f"❌ No videos found in {folder_name}")
        return

    print(f"🚀 Starting incremental clustering for '{folder_name}' ({len(videos)} videos)")
    print("=" * 60)

    init_gemini()
    prompt = load_prompt_v2()

    clusters: List[Cluster] = []
    comparisons_made = []

    for i, video in enumerate(videos):
        print(f"\n🎥 Processing video {i+1}/{len(videos)}: {video.name}")
        
        # If no clusters exist, create the first one
        if not clusters:
            print(f"   ✨ No clusters yet. Creating Cluster 1.")
            clusters.append(Cluster(1, video))
            continue
            
        # Try to match with existing clusters
        matched = False
        for cluster in clusters:
            print(f"   🔍 Comparing with Cluster {cluster.id} (Rep: {cluster.representative.name})...")
            
            # Compare current video with cluster representative
            result = compare_videos(cluster.representative, video, prompt)
            
            # Store comparison result for debugging/audit
            comparisons_made.append({
                "video_1": cluster.representative.name,
                "video_2": video.name,
                "same_group": result.get("same_group"),
                "differences": result.get("differences", [])
            })
            
            if result.get("same_group") is True:
                print(f"   ✅ MATCH! Added to Cluster {cluster.id}")
                cluster.add_video(video)
                matched = True
                break # Greedy match: stop after finding first match
            else:
                print(f"   ❌ Different from Cluster {cluster.id}")
        
        # If no match found after checking all clusters, create a new one
        if not matched:
            new_id = len(clusters) + 1
            print(f"   ✨ No match found. Creating Cluster {new_id}.")
            clusters.append(Cluster(new_id, video))

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
        "total_videos": len(videos),
        "total_clusters": len(clusters),
        "clusters": [c.to_dict() for c in clusters],
        "comparisons_history": comparisons_made
    }

    if not output_file:
        output_file = f"{folder_name}_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Incremental Clustering for A/B Test Videos")
    parser.add_argument("--folder", "-f", required=True, help="Folder containing videos")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    args = parser.parse_args()
    perform_clustering(args.folder, args.output)


if __name__ == "__main__":
    main()
