#!/usr/bin/env python3
"""
Autotest to check stability of cluster_videos.py by running it multiple times.
"""

import argparse
import subprocess
import sys
import re
import time

def parse_clusters_from_output(output):
    """
    Extracts the cluster printout from the stdout of cluster_videos.py.
    Looking for lines starting with 'Cluster X' and following lines.
    """
    lines = output.splitlines()
    cluster_lines = []
    capture = False
    
    # We look for the start of the summary
    for line in lines:
        if "CLUSTERING COMPLETE" in line:
            capture = True
            continue
        
        if capture:
            # Stop if we hit the "Results saved" line or similar end markers if any
            if "Results saved to:" in line:
                break
            # We want to keep lines that look like cluster info
            # The script prints "Cluster X..." and "  - video..." and separator lines
            if line.strip().startswith("Cluster") or line.strip().startswith("-") or line.strip().startswith("="):
                 # Skip the separator line immediately after CLUSTERING COMPLETE
                if "======" in line and not cluster_lines: 
                    continue
                cluster_lines.append(line)

    # Clean up leading/trailing empty lines
    return "\n".join(cluster_lines).strip()

def main():
    parser = argparse.ArgumentParser(description="Run cluster_videos.py multiple times to test stability.")
    parser.add_argument("--folder", "-f", default="Visify", help="Folder to test clustering on")
    parser.add_argument("--runs", "-n", type=int, default=10, help="Number of times to run the test")
    
    args = parser.parse_args()
    
    print(f"Starting stability test for folder '{args.folder}' with {args.runs} runs...\n")

    for i in range(1, args.runs + 1):
        print(f"Run {i} in progress...", file=sys.stderr)
        
        # Build command
        cmd = [sys.executable, "cluster_videos.py", "--folder", args.folder]
        
        try:
            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Extract relevant cluster info
            clusters_text = parse_clusters_from_output(output)
            
            print(f"Run {i}:")
            if clusters_text:
                print(clusters_text)
            else:
                print("(No clusters found or output format changed)")
            print("\n")
            
        except subprocess.CalledProcessError as e:
            print(f"Run {i}: FAILED")
            print(f"Error: {e.stderr}")
            print("\n")
        except Exception as e:
            print(f"Run {i}: EXCEPTION")
            print(f"Error: {str(e)}")
            print("\n")
        
        # Wait 1 minute between runs (except after the last run)
        if i < args.runs:
            print(f"⏳ Waiting 60 seconds before next run...", file=sys.stderr)
            time.sleep(60)

if __name__ == "__main__":
    main()


