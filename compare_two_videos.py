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
import uuid
import hashlib
import threading
import sqlite3
import fcntl
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import List
from dotenv import load_dotenv

from prompts import COMPARISON_PROMPT, JUDGE_PROMPT_TEMPLATE, JUDGE_FOCUSED_PROMPT_TEMPLATE, JUDGE_SIDE_BY_SIDE_PROMPT_TEMPLATE

# New SDK imports
from google import genai
from google.genai import types

# Suppress warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# Base directory
BASE_DIR = Path(__file__).parent

# Global client and settings
_API_KEY = None
_thread_local = threading.local()
RUN_ID = uuid.uuid4().hex[:8]
USE_CACHE_DEFAULT = True
MAX_CONCURRENT_MODEL_CALLS = int(os.getenv("MAX_CONCURRENT_MODEL_CALLS", "4"))
_MODEL_CALL_SEM = threading.Semaphore(MAX_CONCURRENT_MODEL_CALLS)

# Default FPS settings
DEFAULT_ANALYSIS_FPS = 3
DEFAULT_JUDGE_FPS = 3
DEFAULT_FOCUSED_JUDGE_FPS = 5

# Persistent upload cache (local) to avoid expensive remote listing.
UPLOAD_CACHE_DB = BASE_DIR / ".upload_cache.sqlite3"
UPLOAD_CACHE_LOCK_DIR = BASE_DIR / ".upload_cache_locks"


_old_record_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs):
    record = _old_record_factory(*args, **kwargs)
    # Ensure every log record (including third-party libs) has run_id for formatter.
    if not hasattr(record, "run_id"):
        record.run_id = RUN_ID
    return record


logging.setLogRecordFactory(_record_factory)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [run=%(run_id)s] [%(threadName)s] %(message)s"
)

# Silence noisy HTTP client logs (google-genai uses httpx underneath).
for _logger_name in (
    "httpx",
    "httpcore",
    "google",
    "google.auth",
    "google.api_core",
    "google.genai",
    "google.genai.models",
):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


class DropAfcLogsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # google-genai can be chatty about AFC in INFO; drop it.
        # Apply a broad match because the exact message can vary.
        if "AFC" in msg:
            return False
        return True


_afc_filter = DropAfcLogsFilter()
# Attach to handlers (works for propagated third-party logs too).
for _h in logging.getLogger().handlers:
    _h.addFilter(_afc_filter)
# Also attach to known genai loggers in case they have custom handlers.
for _logger_name in ("google.genai", "google.genai.models"):
    logging.getLogger(_logger_name).addFilter(_afc_filter)


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


def build_display_name(video_path: Path, force_unique: bool = False) -> str:
    """Generate a cache-safe display name tied to absolute path and mtime."""
    stat = video_path.stat()
    raw = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    base = f"Video_{video_path.stem}_{digest}"
    if force_unique:
        base = f"{base}_{uuid.uuid4().hex[:8]}"
    return base


def _init_upload_cache() -> None:
    UPLOAD_CACHE_LOCK_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(str(UPLOAD_CACHE_DB)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                display_name TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_uri TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.commit()


def _cache_get(display_name: str):
    if not UPLOAD_CACHE_DB.exists():
        return None
    with sqlite3.connect(str(UPLOAD_CACHE_DB)) as conn:
        row = conn.execute(
            "SELECT file_name, file_uri, mime_type FROM uploads WHERE display_name = ?",
            (display_name,),
        ).fetchone()
    if not row:
        return None
    return {"file_name": row[0], "file_uri": row[1], "mime_type": row[2]}


def _cache_set(display_name: str, file_name: str, file_uri: str, mime_type: str) -> None:
    _init_upload_cache()
    with sqlite3.connect(str(UPLOAD_CACHE_DB)) as conn:
        conn.execute(
            """
            INSERT INTO uploads(display_name, file_name, file_uri, mime_type, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(display_name) DO UPDATE SET
              file_name=excluded.file_name,
              file_uri=excluded.file_uri,
              mime_type=excluded.mime_type,
              updated_at=excluded.updated_at
            """,
            (display_name, file_name, file_uri, mime_type, time.time()),
        )
        conn.commit()


def _cache_delete(display_name: str) -> None:
    if not UPLOAD_CACHE_DB.exists():
        return
    with sqlite3.connect(str(UPLOAD_CACHE_DB)) as conn:
        conn.execute("DELETE FROM uploads WHERE display_name = ?", (display_name,))
        conn.commit()


@contextmanager
def _display_name_lock(display_name: str):
    """
    Cross-process lock to avoid duplicate uploads for the same display_name.
    """
    _init_upload_cache()
    digest = hashlib.sha1(display_name.encode()).hexdigest()
    lock_path = UPLOAD_CACHE_LOCK_DIR / f"{digest}.lock"
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def init_gemini() -> None:
    """Initialize Gemini API credentials (thread-safe client is created lazily)."""
    global _API_KEY
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    _API_KEY = api_key


def get_client():
    """Get a per-thread GenAI client (google-genai client is not guaranteed thread-safe)."""
    if _API_KEY is None:
        init_gemini()
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = genai.Client(api_key=_API_KEY)
        _thread_local.client = client
    return client


def get_or_upload_video_cached(video_path: Path):
    """
    Always-on caching.
    Uses display_name tied to absolute path + mtime to avoid collisions across apps / consoles,
    plus a local sqlite cache to avoid remote listing.
    """
    client = get_client()
    display_name = build_display_name(video_path, force_unique=False)

    with _display_name_lock(display_name):
        cached = _cache_get(display_name)
        if cached:
            try:
                f = client.files.get(name=cached["file_name"])
                while getattr(f, "state", None) == "PROCESSING":
                    print(".", end="", flush=True)
                    time.sleep(2)
                    f = client.files.get(name=f.name)
                if getattr(f, "state", None) == "ACTIVE":
                    return f
            except Exception:
                # Remote file disappeared or became invalid.
                _cache_delete(display_name)

        # Cache miss: try to find by display_name remotely (one-time), then store locally.
        try:
            for f in client.files.list():
                if f.display_name == display_name:
                    while f.state == "PROCESSING":
                        print(".", end="", flush=True)
                        time.sleep(2)
                        f = client.files.get(name=f.name)
                    if f.state == "ACTIVE":
                        _cache_set(display_name, f.name, f.uri, f.mime_type)
                        return f
        except Exception as e:
            logging.warning("Remote cache lookup failed: %s", e)

        # Upload as last resort.
        print(f"📤 Uploading {video_path.name}...")
        try:
            video_file = client.files.upload(
                file=str(video_path),
                config=types.UploadFileConfig(display_name=display_name)
            )

            print(f"⏳ Processing {video_file.name}...", end="", flush=True)
            while video_file.state == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(2)
                video_file = client.files.get(name=video_file.name)

            if video_file.state == "FAILED":
                raise ValueError(f"Video processing failed for {video_path.name}")

            _cache_set(display_name, video_file.name, video_file.uri, video_file.mime_type)
            print(f" ✅ Ready")
            return video_file
        except Exception as e:
            raise ValueError(f"Upload failed: {e}")


def upload_video_ephemeral(video_path: Path):
    """Upload a video without cache reuse (used internally for Judge trims)."""
    client = get_client()
    display_name = build_display_name(video_path, force_unique=True)
    print(f"📤 Uploading {video_path.name} (ephemeral)...")
    try:
        video_file = client.files.upload(
            file=str(video_path),
            config=types.UploadFileConfig(display_name=display_name)
        )

        print(f"⏳ Processing {video_file.name}...", end="", flush=True)
        while video_file.state == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        if video_file.state == "FAILED":
            raise ValueError(f"Video processing failed for {video_path.name}")

        print(f" ✅ Ready")
        return video_file
    except Exception as e:
        raise ValueError(f"Upload failed: {e}")


import shutil
import subprocess
import random
import re

# Global configuration
DOUBLE_CHECK_ENABLED = True
JUDGE_RANDOM_TRIMMING_DEFAULT = False
JUDGE_FOCUSED_TRIM_DEFAULT = True
JUDGE_SIDE_BY_SIDE_DEFAULT = True

def parse_timestamp(time_str: str) -> float:
    """Parse 'MM:SS' or 'SS' string to seconds."""
    try:
        parts = time_str.strip().split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except Exception:
        pass
    return 0.0

def parse_diff_timestamps(diff: dict) -> tuple:
    """
    Extract timestamps from difference object.
    Expected format: "start1 -> end1 | start2 -> end2"
    Returns ((start1, end1), (start2, end2)) in seconds.
    Returns None if parsing fails.
    """
    ts_str = diff.get("timestamp", "")
    if not ts_str:
        return None
    
    try:
        # Split into video 1 and video 2 parts
        parts = ts_str.split('|')
        if len(parts) != 2:
            return None
        
        def parse_range(range_str):
            # "start -> end"
            start_end = range_str.split('->')
            if len(start_end) != 2:
                return 0.0, 0.0
            return parse_timestamp(start_end[0]), parse_timestamp(start_end[1])

        range1 = parse_range(parts[0])
        range2 = parse_range(parts[1])
        
        # Validation: ensure we have non-zero duration or valid start
        if range1 == (0.0, 0.0) and range2 == (0.0, 0.0):
            return None
            
        return range1, range2
    except Exception:
        return None

def trim_video_segment(video_path: Path, start: float, end: float) -> Path:
    """
    Trim video to specific start/end times (in seconds).
    Returns path to temporary trimmed video.
    """
    if not shutil.which("ffmpeg"):
        logging.warning("ffmpeg not found, skipping trim.")
        return video_path

    # Clamp start to 0
    start = max(0.0, start)
    
    random_suffix = hex(random.getrandbits(16))[2:]
    trimmed_path = video_path.parent / f"temp_focus_{random_suffix}_{video_path.name}"
    
    duration = max(1.0, end - start) # Ensure at least 1s duration
    
    # Using -ss before -i is faster and resets timestamp to 0
    # Using -t for duration
    cmd = [
        "ffmpeg", "-y", 
        "-ss", f"{start:.2f}",
        "-i", str(video_path),
        "-t", f"{duration:.2f}",
        "-c", "copy", # Fast copy
        "-avoid_negative_ts", "1",
        str(trimmed_path)
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return trimmed_path
    except subprocess.CalledProcessError:
        logging.warning(f"Failed to trim {video_path} to segment {start}-{end}, using original.")
        return video_path

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

def create_side_by_side_video(video1_path: Path, video2_path: Path) -> Path:
    """
    Stack two videos side-by-side using ffmpeg hstack.
    Returns path to temporary side-by-side video.
    """
    if not shutil.which("ffmpeg"):
        logging.warning("ffmpeg not found, skipping side-by-side.")
        return None
    
    random_suffix = hex(random.getrandbits(16))[2:]
    # Take stem from video1
    sbs_path = video1_path.parent / f"temp_sbs_{random_suffix}_{video1_path.stem}_vs_{video2_path.stem}.mp4"
    
    # We re-encode to ensure compatibility and scale videos to same height to prevent hstack errors.
    # We pick a standard height (e.g. 720p) or just scale both to be same.
    # Simpler approach: Scale both to 720 height, preserving aspect ratio.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video1_path),
        "-i", str(video2_path),
        "-filter_complex", "[0:v]scale=-1:720[v0];[1:v]scale=-1:720[v1];[v0][v1]hstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        str(sbs_path)
    ]
    
    try:
        # Capture output for debugging
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return sbs_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        logging.warning(f"Failed to create side-by-side video. Error: {error_msg}")
        return None

def verify_differences(video1_path: Path, video2_path: Path, differences: list, fps: int = DEFAULT_JUDGE_FPS, random_trimming: bool = True, focused_trim: bool = JUDGE_FOCUSED_TRIM_DEFAULT, focused_fps: int = DEFAULT_FOCUSED_JUDGE_FPS, side_by_side: bool = JUDGE_SIDE_BY_SIDE_DEFAULT) -> tuple[bool, dict]:
    """
    Double-check differences using trimmed videos ("The Judge").
    Returns (is_verified, judge_logs_dict).
    """
    start_time = time.time()
    
    # Check if we should use focused trim (single difference + enabled)
    use_focused_trim = False
    focused_ranges = None
    
    if focused_trim and len(differences) == 1:
        focused_ranges = parse_diff_timestamps(differences[0])
        if focused_ranges:
            use_focused_trim = True

    current_fps = focused_fps if use_focused_trim else fps
    mode_str = "FOCUSED" if use_focused_trim else ("RANDOM_TRIM" if random_trimming else "FULL")
    if use_focused_trim and side_by_side:
        mode_str += "_SBS"
    
    print(f"\n⚖️  The Judge is verifying differences (Mode={mode_str}, FPS={current_fps}, SBS={side_by_side})...")
    
    judge_logs = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode_str,
        "fps": current_fps,
        "verified": None,
        "reasoning": None,
        "raw_response": None
    }
    
    trim1 = None
    trim2 = None
    sbs_video = None

    # 1. Create trimmed versions
    if use_focused_trim and focused_ranges:
        try:
            (start1, end1), (start2, end2) = focused_ranges
            # Add 5 seconds buffer
            trim1 = trim_video_segment(video1_path, start1 - 5, end1 + 5)
            trim2 = trim_video_segment(video2_path, start2 - 5, end2 + 5)
            judge_logs["focused_ranges"] = {"v1": [start1, end1], "v2": [start2, end2]}
            
            # Log exact filenames for user inspection (before deletion)
            judge_logs["trimmed_videos"] = [trim1.name, trim2.name]
            print(f"   ✂️  Focused Judge Videos (temp): {trim1.name}, {trim2.name}")

            # If side-by-side requested, combine them
            if side_by_side:
                sbs_video = create_side_by_side_video(trim1, trim2)
                if sbs_video:
                    judge_logs["sbs_video"] = sbs_video.name
                    print(f"   👯 Side-by-Side Video (temp): {sbs_video.name}")
                else:
                    logging.warning("Side-by-side failed, falling back to separate videos")
                    side_by_side = False # Disable for this run

        except Exception as e:
            logging.warning(f"Focused trim failed: {e}. Falling back to standard method.")
            use_focused_trim = False
            current_fps = fps # Revert FPS
            judge_logs["error_focus_fallback"] = str(e)

    # Fallback / Standard logic if focused trim didn't happen
    if not use_focused_trim:
        if random_trimming:
            trim1 = trim_video_randomly(video1_path)
            trim2 = trim_video_randomly(video2_path)
        else:
            trim1 = video1_path
            trim2 = video2_path
    
    # 2. Upload trimmed videos (ephemeral)
    try:
        if side_by_side and sbs_video:
             # Only upload the SBS video
             v_sbs_file = upload_video_ephemeral(sbs_video)
             v1_file = None
             v2_file = None
        else:
            v1_file = upload_video_ephemeral(trim1)
            v2_file = upload_video_ephemeral(trim2)
    except Exception as e:
        print(f"   ⚠️ Judge upload failed: {e}")
        judge_logs["error"] = str(e)
        return True, judge_logs # Fallback to trusting original result

    # 3. Construct Judge Prompt & Contents
    diff_text = json.dumps(differences, indent=2)
    
    if side_by_side and sbs_video:
        judge_prompt = JUDGE_SIDE_BY_SIDE_PROMPT_TEMPLATE.replace("{differences}", diff_text)
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=judge_prompt),
                    types.Part(text="SIDE-BY-SIDE VIDEO (Left=V1, Right=V2):"),
                    types.Part(
                        file_data=types.FileData(file_uri=v_sbs_file.uri, mime_type=v_sbs_file.mime_type),
                        video_metadata=types.VideoMetadata(fps=current_fps)
                    ),
                ]
            )
        ]
    else:
        if use_focused_trim:
            judge_prompt = JUDGE_FOCUSED_PROMPT_TEMPLATE.replace("{differences}", diff_text)
        else:
            judge_prompt = JUDGE_PROMPT_TEMPLATE.replace("{differences}", diff_text)
            
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=judge_prompt),
                    types.Part(text="VIDEO 1:"),
                    types.Part(
                        file_data=types.FileData(file_uri=v1_file.uri, mime_type=v1_file.mime_type),
                        video_metadata=types.VideoMetadata(fps=current_fps)
                    ),
                    types.Part(text="VIDEO 2:"),
                    types.Part(
                        file_data=types.FileData(file_uri=v2_file.uri, mime_type=v2_file.mime_type),
                        video_metadata=types.VideoMetadata(fps=current_fps)
                    ),
                ]
            )
        ]

    # 4. Call Gemini
    client = get_client()
    try:
        with _MODEL_CALL_SEM:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
        result = json.loads(response.text)
        is_verified = result.get("verified", True) # Default to true if unsure
        
        judge_logs["verified"] = is_verified
        judge_logs["reasoning"] = result.get("reasoning")
        judge_logs["raw_response"] = result
        
        print(f"   🧑‍⚖️ Verdict: {'CONFIRMED' if is_verified else 'REJECTED (Hallucination)'}")
        print(f"   📝 Reasoning: {result.get('reasoning')}")
        return is_verified, judge_logs

    except Exception as e:
        print(f"   ⚠️ Judge error: {e}")
        judge_logs["error"] = str(e)
        return True, judge_logs # Fallback
    finally:
        # Only delete if we created temporary files
        # focused trim ALWAYS creates temps. random trim creates temps.
        # standard non-trim uses originals.
        
        # Helper to safely delete
        def safe_del(p):
            if p and p.exists():
                try:
                    os.remove(p)
                except OSError:
                    pass

        if (use_focused_trim or random_trimming):
            if trim1 and trim1 != video1_path: safe_del(trim1)
            if trim2 and trim2 != video2_path: safe_del(trim2)
            if sbs_video: safe_del(sbs_video)
        
        logging.info(
            "Judge verification finished in %.2f seconds for %s vs %s",
            time.time() - start_time,
            video1_path.name,
            video2_path.name,
        )

def compare_videos(video1_path: Path, video2_path: Path, prompt_text: str = COMPARISON_PROMPT, fps: int = DEFAULT_ANALYSIS_FPS, judge_random_trimming: bool = JUDGE_RANDOM_TRIMMING_DEFAULT, judge_focused_trim: bool = JUDGE_FOCUSED_TRIM_DEFAULT, judge_focused_fps: int = DEFAULT_FOCUSED_JUDGE_FPS, side_by_side: bool = JUDGE_SIDE_BY_SIDE_DEFAULT) -> dict:
    """Compare two videos using Gemini with custom FPS."""
    start_time = time.time()
    print(f"\n🔍 Comparing: {video1_path.name} vs {video2_path.name}")
    
    client = get_client()

    # Get videos (cached or new)
    video1_file = get_or_upload_video_cached(video1_path)
    video2_file = get_or_upload_video_cached(video2_path)
    
    print(f"🤖 Analyzing with Gemini 2.5 Pro (FPS={fps})...")
    
    full_prompt_text = prompt_text
    
    # Construct content with custom FPS
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=full_prompt_text),
                types.Part(text="VIDEO 1 (First Recording):"),
                types.Part(
                    file_data=types.FileData(file_uri=video1_file.uri, mime_type=video1_file.mime_type),
                    video_metadata=types.VideoMetadata(fps=fps)
                ),
                types.Part(text="VIDEO 2 (Second Recording):"),
                types.Part(
                    file_data=types.FileData(file_uri=video2_file.uri, mime_type=video2_file.mime_type),
                    video_metadata=types.VideoMetadata(fps=fps)
                ),
            ]
        )
    ]
    
    # Generate content
    try:
        with _MODEL_CALL_SEM:
            response = client.models.generate_content(
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
        is_real, judge_logs = verify_differences(
            video1_path, 
            video2_path, 
            result.get("differences", []), 
            random_trimming=judge_random_trimming,
            focused_trim=judge_focused_trim,
            focused_fps=judge_focused_fps,
            side_by_side=side_by_side
        )
        
        result["judge_logs"] = judge_logs
        
        if not is_real:
            print("   ✨ Judge overruled: Changing result to SAME GROUP.")
            result["same_group"] = True
            result["differences"] = []
            result["judge_note"] = "Original differences rejected by verification judge."

    result["video_1"] = video1_path.name
    result["video_2"] = video2_path.name
    result["comparison"] = f"{video1_path.stem} vs {video2_path.stem}"
    logging.info(
        "Comparison finished in %.2f seconds for %s vs %s",
        time.time() - start_time,
        video1_path.name,
        video2_path.name,
    )
    
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
    start_time = time.time()
    init_gemini()
    prompt_text = COMPARISON_PROMPT
    
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

    result = compare_videos(v1, v2, prompt_text, fps=args.fps, 
                            judge_random_trimming=args.judge_trim,
                            judge_focused_trim=args.focused_trim,
                            judge_focused_fps=args.focused_fps,
                            side_by_side=args.side_by_side)
    print_result(result)
    
    output_file = args.output or f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f: json.dump(prune_none(result), f, indent=2)
    print(f"\n💾 Saved to: {output_file}")
    logging.info("Compare command finished in %.2f seconds", time.time() - start_time)

def cmd_compare_all(args):
    start_time = time.time()
    folder = BASE_DIR / args.folder
    if not folder.exists(): return print(f"❌ Folder not found: {args.folder}")
    videos = get_videos_in_folder(folder)
    if len(videos) < 2: return print("❌ Need at least 2 videos")
    
    init_gemini()
    prompt_text = COMPARISON_PROMPT
    
    pairs = [(videos[i], videos[(i + 1) % len(videos)]) for i in range(len(videos))]
    print(f"🎥 BATCH COMPARISON: {len(pairs)} pairs in '{args.folder}' (FPS={args.fps}, JudgeTrim={args.judge_trim}, FocusTrim={args.focused_trim}, SBS={args.side_by_side})")
    
    results = []
    for i, (v1, v2) in enumerate(pairs, 1):
        print(f"\n{'#'*60}\n# PAIR {i}/{len(pairs)}\n{'#'*60}")
        result = compare_videos(v1, v2, prompt_text, fps=args.fps, 
                                judge_random_trimming=args.judge_trim,
                                judge_focused_trim=args.focused_trim,
                                judge_focused_fps=args.focused_fps,
                                side_by_side=args.side_by_side)
        results.append(result)
        print_result(result)
        
    unified = {
        "folder": args.folder,
        "timestamp": datetime.now().isoformat(),
        "pipeline_seconds": round(time.time() - start_time, 3),
        "summary": {
            "total": len(results),
            "same": sum(1 for r in results if r.get("same_group") is True),
            "different": sum(1 for r in results if r.get("same_group") is False)
        },
        "comparisons": results
    }
    unified = prune_none(unified)
    
    output_file = args.output or f"{args.folder}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f: json.dump(unified, f, indent=2)
    print(f"\n💾 Saved batch results to: {output_file}")
    logging.info("Compare-all command finished in %.2f seconds", time.time() - start_time)


def main():
    parser = argparse.ArgumentParser(description="A/B Test Video Comparator")
    subparsers = parser.add_subparsers(dest="command")
    start_time = time.time()
    
    list_p = subparsers.add_parser("list")
    list_p.add_argument("--folder", "-f")
    
    comp_p = subparsers.add_parser("compare")
    comp_p.add_argument("video1")
    comp_p.add_argument("video2")
    comp_p.add_argument("--folder", "-f")
    comp_p.add_argument("--output", "-o")
    comp_p.add_argument("--fps", type=int, default=DEFAULT_ANALYSIS_FPS, help="Frames per second for analysis")
    comp_p.add_argument("--no-judge-trim", action="store_false", dest="judge_trim", default=True, help="Disable random trimming for judge verification")
    comp_p.add_argument("--no-focused-trim", action="store_false", dest="focused_trim", default=True, help="Disable focused trimming for single difference")
    comp_p.add_argument("--focused-fps", type=int, default=DEFAULT_FOCUSED_JUDGE_FPS, help="FPS for focused judge analysis")
    comp_p.add_argument("--no-side-by-side", action="store_false", dest="side_by_side", default=True, help="Disable side-by-side video for focused judge")

    
    all_p = subparsers.add_parser("compare-all")
    all_p.add_argument("--folder", "-f", required=True)
    all_p.add_argument("--output", "-o")
    all_p.add_argument("--fps", type=int, default=DEFAULT_ANALYSIS_FPS, help="Frames per second for analysis")
    all_p.add_argument("--no-judge-trim", action="store_false", dest="judge_trim", default=True, help="Disable random trimming for judge verification")
    all_p.add_argument("--no-focused-trim", action="store_false", dest="focused_trim", default=True, help="Disable focused trimming for single difference")
    all_p.add_argument("--focused-fps", type=int, default=DEFAULT_FOCUSED_JUDGE_FPS, help="FPS for focused judge analysis")
    all_p.add_argument("--no-side-by-side", action="store_false", dest="side_by_side", default=True, help="Disable side-by-side video for focused judge")

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "compare-all":
        cmd_compare_all(args)
    else:
        parser.print_help()
        return

    logging.info("Total runtime: %.2f seconds", time.time() - start_time)

if __name__ == "__main__":
    main()
