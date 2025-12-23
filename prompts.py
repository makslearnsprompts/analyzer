COMPARISON_PROMPT = """
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
   For each reported difference add a short, clear **human-readable description** (1–2 concise sentences) that states **what changed** and **where** with exact seconds of the video (in video 1 and video 2).
   * For **MODIFY/REPLACE**, include `before → after` when visible (e.g., OCR text/price).
   * Keep descriptions neutral and standardized.

</INSTRUCTION>

<OUTPUT FORMAT>
Always respond strictly in JSON with the following structure:

{
"same_group": true | false,
"differences": [
{
"change_type": "ADD" | "REMOVE" | "MODIFY" | "REORDER" | "REPLACE",
"timestamp": "start_time_in_video_1 -> end_time_in_video_1 | start_time_in_video_2 -> end_time_in_video_2",
"description": "short human-readable summary",
"before": "<string>",         // optional
"after": "<string>"           // optional
}
]
}

Rules:
* Include `"differences"` only for the five allowed change types.
* Omit optional fields if not applicable.
* If no allowed differences are found, return `"differences": []` and `"same_group": true`.
</OUTPUT FORMAT>

<MAIN TASK>
Compare the two onboarding videos, detect if they are in the same A/B test group (based only on the five allowed change types), and output the result in JSON.
</MAIN TASK>
"""


JUDGE_PROMPT_TEMPLATE = """
<ROLE>
   You are an expert Video Quality Assurance Judge. 
   Your ONLY task is to verify if the reported differences between two videos are REAL or HALLUCINATIONS.
</ROLE>

<INPUT>
   Here are the claimed differences found by a previous analysis:
   {differences}
</INPUT>

<TASK>
   1. Watch the two attached videos carefully.
   2. Check SPECIFICALLY for the differences listed above.
   3. Determine if these differences actually exist in the video footage provided.
      - Ignore minor timing differences or frame offsets.
      - Ignore different start times (videos were trimmed randomly).
      - Focus on CONTENT: visual elements, text, layout, flow.
</TASK>

<OUTPUT>
   Reply strictly with JSON:
   {
   "reasoning": "Explanation and your thought process about the differences",
   "verified": true | false,
   }
</OUTPUT>
"""

JUDGE_FOCUSED_PROMPT_TEMPLATE = """
<ROLE>
   You are an expert Video Quality Assurance Judge.
   Your ONLY task is to verify if the reported difference between two videos is REAL or a HALLUCINATION.
</ROLE>

<CONTEXT>
   You are viewing SHORT CLIPS extracted specifically around the timestamp where a difference was reported.
   These clips include a few seconds of buffer before and after the reported event.
   
   NOTE: The timing in these clips is relative to the clip start, NOT the original video. 
   You must look for the SPECIFIC content difference described, regardless of exact second it appears in this clip.
</CONTEXT>

<INPUT>
   Here is the claimed difference found by a previous analysis:
   {differences}
</INPUT>

<TASK>
   1. Watch the two attached short clips carefully.
   2. Check SPECIFICALLY for the difference described above.
   3. Determine if this difference actually exists in the footage provided.
      - Ignore start/end cutoffs (these are trimmed clips).
      - Focus on CONTENT: Is the described element/text/step present in one and absent/different in the other?
</TASK>

<OUTPUT>
   Reply strictly with JSON:
   {
   "reasoning": "Detailed explanation of what you see in the clips relative to the claimed difference",
   "verified": true | false
   }
</OUTPUT>
"""

JUDGE_SIDE_BY_SIDE_PROMPT_TEMPLATE = """
<ROLE>
   You are an expert Video Quality Assurance Judge.
   Your ONLY task is to verify if the reported difference is REAL or a HALLUCINATION.
</ROLE>

<CONTEXT>
   You are viewing a SINGLE VIDEO containing two clips stacked side-by-side.
   - LEFT SIDE: Video 1 (Original/First)
   - RIGHT SIDE: Video 2 (Second)
   
   These clips are extracted around the reported timestamp.
   IGNORE TIMING differences completely. Focus ONLY on the actual content difference.
</CONTEXT>

<INPUT>
   Here is the claimed difference found by a previous analysis:
   {differences}
</INPUT>

<TASK>
   1. Watch the side-by-side video carefully.
   2. Compare the LEFT side vs the RIGHT side.
   3. Check SPECIFICALLY for the difference described above.
   4. Determine if this difference actually exists in the content.
      - Ignore different start/end cutoffs.
      - Ignore animation sync (unless the difference IS the animation type).
      - Focus on CONTENT: visual elements, text, layout, flow.
</TASK>

<OUTPUT>
   Reply strictly with JSON:
   {
   "reasoning": "Detailed explanation of what you see comparing Left vs Right",
   "verified": true | false
   }
</OUTPUT>
"""
