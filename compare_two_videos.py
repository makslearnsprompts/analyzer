#!/usr/bin/env python3
"""
CLI tool to compare videos for A/B test detection using Gemini (Google GenAI SDK).
Features:
- Caches uploaded videos to avoid re-uploading
- Compares video pairs for A/B differences
- Uses custom FPS sampling for better video analysis
"""

import argparse
import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
from dotenv import load_dotenv

# New SDK imports
from google import genai
from google.genai import types

# Suppress warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# Base directory
BASE_DIR = Path(__file__).parent

# Global client
CLIENT = None

def load_prompt_v2() -> str:
    """Load the V2 comparison prompt."""
    prompt_file = BASE_DIR / "prompt.py"
    with open(prompt_file, 'r') as f:
        content = f.read()
    
    start = content.find('PROMPT_FOR_FIND_DIFFERENCE_V2 = """') + len('PROMPT_FOR_FIND_DIFFERENCE_V2 = """')
    end = content.find('"""', start)
    return content[start:end].strip()


def init_gemini() -> None:
    """Initialize Gemini API Client."""
    global CLIENT
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    # Initialize the client
    CLIENT = genai.Client(api_key=api_key)


def get_or_upload_video(video_path: Path):
    """
    Get existing uploaded file or upload new one using new SDK.
    Uses display_name to identify if file was already uploaded.
    """
    if CLIENT is None:
        init_gemini()
        
    display_name = f"Video_{video_path.stem}"
    
    # Check existing files
    print(f"🔎 Checking cache for {video_path.name}...")
    try:
        # Note: client.files.list returns an iterator
        for f in CLIENT.files.list():
            if f.display_name == display_name:
                print(f"   Found cached file: {f.name}")
                
                # Wait if still processing
                while f.state == "PROCESSING":
                    print(".", end="", flush=True)
                    time.sleep(2)
                    f = CLIENT.files.get(name=f.name)
                
                if f.state == "ACTIVE":
                    print("   ✅ Ready (Cached)")
                    return f
                else:
                    print(f"   ⚠️ Cached file state is {f.state}, re-uploading...")
    except Exception as e:
        print(f"   ⚠️ Error checking cache: {e}")

    # Upload if not found or not active
    print(f"📤 Uploading {video_path.name}...")
    try:
        video_file = CLIENT.files.upload(
            file=str(video_path),
            config=types.UploadFileConfig(display_name=display_name)
        )
        
        print(f"⏳ Processing {video_file.name}...", end="", flush=True)
        while video_file.state == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            video_file = CLIENT.files.get(name=video_file.name)
        
        if video_file.state == "FAILED":
            raise ValueError(f"Video processing failed for {video_path.name}")
        
        print(f" ✅ Ready")
        return video_file
    except Exception as e:
        raise ValueError(f"Upload failed: {e}")


import shutil
import subprocess
import random

# Global configuration
DOUBLE_CHECK_ENABLED = True

def trim_video_randomly(video_path: Path) -> Path:
    """
    Trim video randomly (start 0.1-0.5s) to force new hash/frames.
    Returns path to temporary trimmed video.
    """
    if not shutil.which("ffmpeg"):
        logging.warning("ffmpeg not found, skipping trim.")
        return video_path

    random_start = random.uniform(0.1, 0.5)
    random_suffix = hex(random.getrandbits(16))[2:]
    trimmed_path = video_path.parent / f"temp_trim_{random_suffix}_{video_path.name}"
    
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ss", f"{random_start:.2f}",
        "-c", "copy",
        str(trimmed_path)
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return trimmed_path
    except subprocess.CalledProcessError:
        logging.warning(f"Failed to trim {video_path}, using original.")
        return video_path

def verify_differences(video1_path: Path, video2_path: Path, differences: list) -> bool:
    """
    Double-check differences using trimmed videos ("The Judge").
    Returns True if differences are confirmed, False if hallucination.
    """
    print(f"\n⚖️  The Judge is verifying differences...")
    
    # 1. Create trimmed versions
    trim1 = trim_video_randomly(video1_path)
    trim2 = trim_video_randomly(video2_path)
    
    # 2. Upload trimmed videos (force new upload due to unique name)
    try:
        v1_file = get_or_upload_video(trim1)
        v2_file = get_or_upload_video(trim2)
    except Exception as e:
        print(f"   ⚠️ Judge upload failed: {e}")
        return True # Fallback to trusting original result

    # 3. Construct Judge Prompt
    diff_text = json.dumps(differences, indent=2)
    judge_prompt = f"""
<ROLE>
You are an expert Video Quality Assurance Judge. 
Your ONLY task is to verify if the reported differences between two videos are REAL or HALLUCINATIONS.
</ROLE>

<INPUT>
Here are the claimed differences found by a previous analysis:
{diff_text}
</INPUT>

<TASK>
1. Watch the two attached videos carefully.
2. Check SPECIFICALLY for the differences listed above.
3. Determine if these differences actually exist in the video footage provided.
   - Ignore minor timing differences or frame offsets.
   - Ignore different start times (videos were trimmed randomly).
   - Focus on CONTENT: visual elements, text, layout, flow.

Reply strictly with JSON:
{{
  "verified": true, // true if AT LEAST ONE meaningful difference exists
  "reasoning": "Explanation of why you confirmed or rejected the differences"
}}
</TASK>
"""

    # 4. Call Gemini
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=judge_prompt),
                types.Part(text="VIDEO 1:"),
                types.Part(
                    file_data=types.FileData(file_uri=v1_file.uri, mime_type=v1_file.mime_type),
                    video_metadata=types.VideoMetadata(fps=5)
                ),
                types.Part(text="VIDEO 2:"),
                types.Part(
                    file_data=types.FileData(file_uri=v2_file.uri, mime_type=v2_file.mime_type),
                    video_metadata=types.VideoMetadata(fps=5)
                ),
            ]
        )
    ]

    try:
        response = CLIENT.models.generate_content(
            model="gemini-2.5-pro",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        result = json.loads(response.text)
        is_verified = result.get("verified", True) # Default to true if unsure
        print(f"   🧑‍⚖️ Verdict: {'CONFIRMED' if is_verified else 'REJECTED (Hallucination)'}")
        print(f"   📝 Reasoning: {result.get('reasoning')}")
        
        # Cleanup temp files
        if trim1 != video1_path: os.remove(trim1)
        if trim2 != video2_path: os.remove(trim2)
        
        return is_verified

    except Exception as e:
        print(f"   ⚠️ Judge error: {e}")
        return True # Fallback

def compare_videos(video1_path: Path, video2_path: Path, prompt: str) -> dict:
    """Compare two videos using Gemini with custom FPS."""
    print(f"\n🔍 Comparing: {video1_path.name} vs {video2_path.name}")
    
    if CLIENT is None:
        init_gemini()

    # Get videos (cached or new)
    video1_file = get_or_upload_video(video1_path)
    video2_file = get_or_upload_video(video2_path)
    
    print(f"🤖 Analyzing with Gemini 2.5 Pro (FPS=5)...")
    
    full_prompt_text = f"""
<INSTRUCTION>
You are analyzing **two silent screen recordings** of the onboarding process of the same iOS application.
Each video was recorded by a different agent while going through the onboarding flow.

Your main goal:

1. Decide whether the two videos belong to the **same A/B test group** or to **different groups**.

2. Compare them screen-by-screen and capture only **meaningful differences** that fit exactly one of the following change types (STRICT ENUM):

   * **ADD** — an element/step/screen was added.
   * **REMOVE** — an element/step/screen was removed.
   * **MODIFY** — a STATIC attribute changed (copy/text, static color, static size, CTA text, price, trial_days, order of bullets).
   * **REORDER** — the order of elements/steps changed.
   * **REPLACE** — one element was replaced by a conceptually different one (e.g., icon → image).

3. **CRITICAL EXCLUSIONS (Ignore these completely):**
   * **ANIMATIONS & LOOPS:** Treat all animations, videos, and dynamic graphics as **"Black Boxes"**. Do NOT analyze the internal state, specific frame, number of items currently visible in a loop, or which element is currently highlighted within an animation. If the *subject* of the animation is the same (e.g., both show a "scanning" concept), treat them as IDENTICAL, even if they are out of sync or show different phases.
   * **TIMING:** Animation speed, duration, or frame synchronization.
   * **SYSTEM:** Skeleton loaders, network delays, status bar (battery/wifi/time), OS visual style, notifications, cursor/touch indicators.
   * **USER BEHAVIOR:** User navigating at different speeds or choosing different options when the same options are available.
   * **RESPONSIVENESS:** Layout reflows due to different screen sizes.

4. Decision rule:
   * If you detect **at least one** difference of the allowed types above → `"same_group": false`.
   * If you detect **none** of the allowed types → `"same_group": true`.

5. **Comparison Logic:**
   * Focus on **SEMANTIC** identity, not pixel-perfect identity.
   * For animations: Ask yourself, "Is this the same asset playing at a different time?" If yes → **IGNORE**. Only report if the asset itself is fundamentally different (e.g., a broom animation vs. a vacuum cleaner animation).

6. Text requirement:
   For each reported difference add a short, clear **human-readable description** (1–2 concise sentences) that states **what changed** and **where**.
   * For **MODIFY/REPLACE**, include `before → after` when visible (e.g., OCR text/price).
   * Keep descriptions neutral and standardized.

</INSTRUCTION>

<OUTPUT FORMAT>
Always respond strictly in JSON with the following structure:

{{
"same_group": true | false,
"differences": [
{{
"change_type": "ADD" | "REMOVE" | "MODIFY" | "REORDER" | "REPLACE",
"description": "short human-readable summary",
"before": "<string>",         // optional
"after": "<string>"           // optional
}}
]
}}

Rules:
* Include `"differences"` only for the five allowed change types.
* Omit optional fields if not applicable.
* If no allowed differences are found, return `"differences": []` and `"same_group": true`.
</OUTPUT FORMAT>

<MAIN TASK>
Compare the two onboarding videos, detect if they are in the same A/B test group (based only on the five allowed change types), and output the result in JSON.
</MAIN TASK>

{prompt}
"""

    # Construct content with custom FPS
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=full_prompt_text),
                types.Part(text="VIDEO 1 (First Recording):"),
                types.Part(
                    file_data=types.FileData(file_uri=video1_file.uri, mime_type=video1_file.mime_type),
                    video_metadata=types.VideoMetadata(fps=5)
                ),
                types.Part(text="VIDEO 2 (Second Recording):"),
                types.Part(
                    file_data=types.FileData(file_uri=video2_file.uri, mime_type=video2_file.mime_type),
                    video_metadata=types.VideoMetadata(fps=5)
                ),
            ]
        )
    ]
    
    # Generate content
    try:
        response = CLIENT.models.generate_content(
            model="gemini-2.5-pro",
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        response_text = response.text.strip()
        
        print(f"📋 Raw Response:\n{response_text}\n")
        
        # Parse JSON
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            result = json.loads(response_text[json_start:json_end])
        else:
            result = {"same_group": None, "differences": [], "error": "Could not parse JSON", "raw": response_text}
            
    except Exception as e:
        result = {"same_group": None, "differences": [], "error": str(e)}

    # === DOUBLE CHECK / JUDGE MECHANISM ===
    if DOUBLE_CHECK_ENABLED and result.get("same_group") is False:
        is_real = verify_differences(video1_path, video2_path, result.get("differences", []))
        if not is_real:
            print("   ✨ Judge overruled: Changing result to SAME GROUP.")
            result["same_group"] = True
            result["differences"] = []
            result["judge_note"] = "Original differences rejected by verification judge."

    result["video_1"] = video1_path.name
    result["video_2"] = video2_path.name
    result["comparison"] = f"{video1_path.stem} vs {video2_path.stem}"
    
    return result


def print_result(result: dict) -> None:
    """Print comparison result."""
    print(f"\n{'='*60}")
    print(f"📊 RESULT: {result.get('comparison', 'Unknown')}")
    print(f"{'='*60}")
    
    same_group = result.get("same_group")
    if same_group is True:
        print(f"✅ SAME GROUP (no A/B differences)")
    elif same_group is False:
        print(f"🔄 DIFFERENT GROUPS (A/B test detected)")
    else:
        print(f"❓ UNCLEAR")
    
    if "error" in result:
        print(f"⚠️  Error: {result['error']}")
    
    differences = result.get("differences", [])
    if differences:
        print(f"\n🔍 Differences ({len(differences)}):")
        for i, diff in enumerate(differences, 1):
            if isinstance(diff, dict):
                desc = diff.get('description', diff.get('element', str(diff)))
                print(f"   {i}. {desc}")
            else:
                print(f"   {i}. {diff}")
    print(f"{'='*60}")


# ============== CLI Helpers ==============

def get_video_folders() -> List[Path]:
    return sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and not d.name.startswith('.') and list(d.glob("*.mp4"))])

def get_videos_in_folder(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.mp4"))

def cmd_list(args):
    if args.folder:
        folder = BASE_DIR / args.folder
        if not folder.exists(): return print(f"❌ Folder not found: {args.folder}")
        videos = get_videos_in_folder(folder)
        print(f"\n📁 Videos in '{args.folder}' ({len(videos)} total):\n")
        for i, v in enumerate(videos, 1): print(f"   {i}. {v.name}")
    else:
        folders = get_video_folders()
        print(f"\n📂 Available folders:\n")
        for f in folders: print(f"   • {f.name}/ ({len(get_videos_in_folder(f))} videos)")

def cmd_compare(args):
    init_gemini()
    prompt = load_prompt_v2()
    
    if args.folder:
        folder = BASE_DIR / args.folder
        if not folder.exists(): return print(f"❌ Folder not found: {args.folder}")
        videos = get_videos_in_folder(folder)
        try:
            v1 = videos[int(args.video1)-1]
            v2 = videos[int(args.video2)-1]
        except (ValueError, IndexError):
            return print("❌ Invalid video indices")
    else:
        v1, v2 = Path(args.video1), Path(args.video2)
        if not v1.is_absolute(): v1 = BASE_DIR / v1
        if not v2.is_absolute(): v2 = BASE_DIR / v2

    if not v1.exists() or not v2.exists(): return print("❌ Video file not found")

    result = compare_videos(v1, v2, prompt)
    print_result(result)
    
    output_file = args.output or f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f: json.dump(result, f, indent=2)
    print(f"\n💾 Saved to: {output_file}")

def cmd_compare_all(args):
    folder = BASE_DIR / args.folder
    if not folder.exists(): return print(f"❌ Folder not found: {args.folder}")
    videos = get_videos_in_folder(folder)
    if len(videos) < 2: return print("❌ Need at least 2 videos")
    
    init_gemini()
    prompt = load_prompt_v2()
    
    pairs = [(videos[i], videos[(i + 1) % len(videos)]) for i in range(len(videos))]
    print(f"🎥 BATCH COMPARISON: {len(pairs)} pairs in '{args.folder}'")
    
    results = []
    for i, (v1, v2) in enumerate(pairs, 1):
        print(f"\n{'#'*60}\n# PAIR {i}/{len(pairs)}\n{'#'*60}")
        result = compare_videos(v1, v2, prompt)
        results.append(result)
        print_result(result)
        
    unified = {
        "folder": args.folder,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "same": sum(1 for r in results if r.get("same_group") is True),
            "different": sum(1 for r in results if r.get("same_group") is False)
        },
        "comparisons": results
    }
    
    output_file = args.output or f"{args.folder}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f: json.dump(unified, f, indent=2)
    print(f"\n💾 Saved batch results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="A/B Test Video Comparator")
    subparsers = parser.add_subparsers(dest="command")
    
    list_p = subparsers.add_parser("list")
    list_p.add_argument("--folder", "-f")
    
    comp_p = subparsers.add_parser("compare")
    comp_p.add_argument("video1")
    comp_p.add_argument("video2")
    comp_p.add_argument("--folder", "-f")
    comp_p.add_argument("--output", "-o")
    
    all_p = subparsers.add_parser("compare-all")
    all_p.add_argument("--folder", "-f", required=True)
    all_p.add_argument("--output", "-o")
    
    args = parser.parse_args()
    if args.command == "list": cmd_list(args)
    elif args.command == "compare": cmd_compare(args)
    elif args.command == "compare-all": cmd_compare_all(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
