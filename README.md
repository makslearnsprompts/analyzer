# A/B Test Video Analyzer

This project provides tools to analyze and cluster iOS app onboarding screen recordings to detect A/B tests. It uses Google's Gemini 2.5 Pro model to compare videos semantically, ignoring minor timing or artifacts, and focusing on structural, copy, or visual differences.

## Features

- **Video Comparison**: Compare two videos to determine if they belong to the same A/B test group.
- **Incremental Clustering**: Automatically group a folder of videos into clusters based on their A/B test variant.
- **"The Judge" Verification**: A double-check mechanism that verifies detected differences by re-analyzing trimmed versions of the videos to reduce hallucinations.
- **Smart Caching**: Avoids re-uploading videos to Gemini if they have already been processed.

## Prerequisites

- **Python 3.8+**
- **FFmpeg**: Required for the "Judge" verification feature (video trimming).
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
- **Gemini API Key**: You need a valid API key from Google AI Studio.

## Installation

1.  Clone the repository.
2.  Install Python dependencies:
    ```bash
    pip install google-genai python-dotenv
    ```
3.  Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_api_key_here
    ```

## Usage

### 1. Compare Two Videos

Compare two specific video files to see if they are different.

```bash
python3 compare_two_videos.py compare path/to/video1.mp4 path/to/video2.mp4
```

**Options:**
- `--folder` / `-f`: Specify a folder relative to the project root (optional).
- `--output` / `-o`: Specify an output JSON file path.

**Example:**
```bash
python3 compare_two_videos.py compare video_A.mp4 video_B.mp4 -o result.json
```

### 2. Cluster Videos

Analyze a folder containing multiple videos and group them into clusters where each cluster represents a unique A/B test variant.

```bash
python3 cluster_videos.py --folder folder_name
```

**How it works:**
1.  Takes the first video as the representative of Cluster 1.
2.  Compares subsequent videos against existing cluster representatives.
3.  If a match is found (`same_group: true`), the video is added to that cluster.
4.  If no match is found, a new cluster is created.
5.  Results are saved to a JSON file (e.g., `folder_clustering_TIMESTAMP.json`).

### 3. List Available Videos

List all video folders or videos within a specific folder.

```bash
# List all folders with videos
python3 compare_two_videos.py list

# List videos in a specific folder
python3 compare_two_videos.py list --folder folder_name
```

## Output Format

The tools output JSON results containing:
- `same_group`: Boolean indicating if videos are from the same variant.
- `differences`: List of detected changes (ADD, REMOVE, MODIFY, REORDER, REPLACE).
- `reasoning`: (Internal) Logic used by the model.

Example JSON Output:
```json
{
  "same_group": false,
  "differences": [
    {
      "change_type": "MODIFY",
      "description": "The CTA button color changed.",
      "before": "Blue",
      "after": "Green"
    }
  ],
  "video_1": "v1.mp4",
  "video_2": "v2.mp4"
}
```

## Project Structure

- `compare_two_videos.py`: Main CLI tool and core logic for video comparison.
- `cluster_videos.py`: Script for clustering multiple videos.
- `prompt.py`: Contains the system prompts used for Gemini analysis.
