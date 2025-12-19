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
   For each reported difference add a short, clear **human-readable description** (1–2 concise sentences) that states **what changed** and **where**.
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

<REFERENCE_PROMPT_V2>
<INSTRUCTION>
You are analyzing **two silent screen recordings** of the onboarding process of the same iOS application.
Each video was recorded by a different agent while going through the onboarding flow.

Your main goal:

1. Decide whether the two videos belong to the **same A/B test group** or to **different groups**.

2. Compare them screen-by-screen and capture only **meaningful differences** that fit exactly one of the following change types (STRICT ENUM):

   * **ADD** — an element/step/screen was added.
   * **REMOVE** — an element/step/screen was removed.
   * **MODIFY** — an attribute changed (copy/text, color, size, CTA text, price, trial_days, order of bullets, etc.).
   * **REORDER** — the order of elements/steps changed.
   * **REPLACE** — one element was replaced by another (e.g., icon → image).

3. **Ignore everything else** (do NOT treat as A/B):
   timing/animation speed (very important!!! There there is difference in timing of animations, it means problem on the side of video), skeleton loaders, network delays, device/screen-size responsive reflow without role change, status bar (battery/wifi/time), notifications, cursor movement, OS visual style differences, user choosing a different path when the same options are visible in both versions.

4. Decision rule:

   * If you detect **at least one** difference of the allowed types above → `"same_group": false`.
   * If you detect **none** of the allowed types → `"same_group": true`.

5. The bot agents may navigate differently (delays, pressing different visible options). Be careful to distinguish user choice from real A/B differences. Only report differences that clearly fit the allowed types.

6. Text requirement:
   For each reported difference add a short, clear **human-readable description** (1–2 concise sentences) that states **what changed** and **where** (e.g., screen or element).

   * For **MODIFY/REPLACE**, include `before → after` when visible (e.g., OCR text/price).
   * Keep descriptions neutral and standardized (no speculation).

</INSTRUCTION>

<OUTPUT FORMAT>
Always respond strictly in JSON with the following structure:

{
"same_group": true | false,
"differences": [
{
"change_type": "ADD" | "REMOVE" | "MODIFY" | "REORDER" | "REPLACE",
"description": "short human-readable summary of what changed and where",
"before": "<string>",         // optional: previous value/text/asset for MODIFY/REPLACE
"after": "<string>"           // optional: new value/text/asset for MODIFY/REPLACE
}
]
}

Rules:

* Include `"differences"` only for the five allowed change types.
* Omit optional fields if not applicable or unknown.
* If no allowed differences are found, return `"differences": []` and `"same_group": true`.

</OUTPUT FORMAT>

<MAIN TASK>
Compare the two onboarding videos, detect if they are in the same A/B test group (based only on the five allowed change types), and output the result in JSON.

<MOTIVATION>
Your accuracy is critical for detecting A/B tests. Be precise, exhaustive within the five types, and avoid assumptions outside of them.
</MOTIVATION>
</REFERENCE_PROMPT_V2>
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
