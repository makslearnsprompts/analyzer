
PROMPT_FOR_FIND_DIFFERENCE_V3 = """
<INSTRUCTION>
You are analyzing **two silent screen recordings** of the onboarding process of the same iOS application to detect A/B tests.
Each video was recorded by a different agent while going through the onboarding flow.

Your Goal:
Compare them screen-by-screen and capture only **structural or copy differences** (A/B tests).

<CRITICAL_RULES_FOR_VIDEO_PROCESSING>
Since you are processing video frames, you must handle **Sampling Artifacts** and **Animation Phases** intelligently:

1.  **THE "MULTI-STAGE ANIMATION" RULE (Ignore Phase Mismatch):**
    Many onboarding animations are sequences (e.g., An icon of a *Trash Can* transforms into *Arrows/Chevrons*, or a *Broom* moves across files).
    *   **Constraint:** If Video A shows "Symbol X" and Video B shows "Symbol Y" in the *same context/screen*, you must assume this is just a **different timestamp of the SAME animation**.
    *   **Action:** Do NOT report this as REPLACE or MODIFY. Treat the entire animation block as a single "Black Box".
    *   **Exception:** Only report if the *theme* changes entirely (e.g., "Cleaning Animation" vs "Rocket Launch Animation").

2.  **THE "FAST USER" RULE (Transient Screens):**
    Agents click at different speeds. In one video, a screen (like a Paywall or Loader) might appear for 0.5 seconds; in another, for 5 seconds.
    *   **Constraint:** If a screen appears *even for a split second* in one video, it counts as "EXISTING".
    *   **Action:** Do NOT report "REMOVE" just because a user clicked through a screen quickly. Only report "REMOVE" if the screen is **structurally impossible to reach** or completely absent from the flow logic.

</CRITICAL_RULES_FOR_VIDEO_PROCESSING>

<ALLOWED_CHANGE_TYPES>
Report differences ONLY if they fit these categories strictly:

*   **ADD** — an element/step/screen exists in one but is completely absent in the other (check "Fast User" rule first!).
*   **REMOVE** — an element/step/screen is completely gone (check "Fast User" rule first!).
*   **MODIFY** — A STATIC attribute changed (Color of a toggle/button, Text copy, Price, Trial duration).
    *   *Note:* Do not report "Animation changed" if it's just a timing difference.
*   **REORDER** — The sequence of screens changed.
*   **REPLACE** — A static asset was swapped for a conceptually different one (e.g., Photo A vs Photo B).
</ALLOWED_CHANGE_TYPES>

<OUTPUT_FORMAT>
Return JSON only:
{
  "same_group": true | false,
  "differences": [
    {
      "change_type": "ADD" | "REMOVE" | "MODIFY" | "REORDER" | "REPLACE",
      "description": "Short description. For MODIFY include colors/text.",
      "before": "Blue toggle",
      "after": "Green toggle"
    }
  ]
}

If no valid differences found, return "differences": [] and "same_group": true.
</OUTPUT_FORMAT>

<MAIN_TASK>
Compare the two videos.
1. First, check if "different" animations are actually just **phases** of the same sequence (Trash can vs Chevrons). If so, IGNORE.
2. Check if "missing" screens were just skipped quickly. If so, IGNORE.
3. Output the JSON.
</MAIN_TASK>
"""



# PROMPT_FOR_FIND_DIFFERENCE_V3 = """
# <INSTRUCTION>
# You are analyzing **two silent screen recordings** of the onboarding process of the same iOS application.
# Each video was recorded by a different agent while going through the onboarding flow.

# Your main goal:

# 1. Decide whether the two videos belong to the **same A/B test group** or to **different groups**.

# 2. Compare them screen-by-screen and capture only **meaningful differences** that fit exactly one of the following change types (STRICT ENUM):

#    * **ADD** — an element/step/screen was added.
#    * **REMOVE** — an element/step/screen was removed.
#    * **MODIFY** — a STATIC attribute changed (copy/text, static color, static size, CTA text, price, trial_days, order of bullets).
#    * **REORDER** — the order of elements/steps changed.
#    * **REPLACE** — one element was replaced by a conceptually different one (e.g., icon → image).

# 3. **CRITICAL EXCLUSIONS (Ignore these completely):**
#    * **ANIMATIONS & LOOPS:** Treat all animations, videos, and dynamic graphics as **"Black Boxes"**. Do NOT analyze the internal state, specific frame, number of items currently visible in a loop, or which element is currently highlighted within an animation. If the *subject* of the animation is the same (e.g., both show a "scanning" concept), treat them as IDENTICAL, even if they are out of sync or show different phases.
#    * **TIMING:** Animation speed, duration, or frame synchronization.
#    * **SYSTEM:** Skeleton loaders, network delays, status bar (battery/wifi/time), OS visual style, notifications, cursor/touch indicators.
#    * **USER BEHAVIOR:** User navigating at different speeds or choosing different options when the same options are available.
#    * **RESPONSIVENESS:** Layout reflows due to different screen sizes.

# 4. Decision rule:
#    * If you detect **at least one** difference of the allowed types above → `"same_group": false`.
#    * If you detect **none** of the allowed types → `"same_group": true`.

# 5. **Comparison Logic:**
#    * Focus on **SEMANTIC** identity, not pixel-perfect identity.
#    * For animations: Ask yourself, "Is this the same asset playing at a different time?" If yes → **IGNORE**. Only report if the asset itself is fundamentally different (e.g., a broom animation vs. a vacuum cleaner animation).

# 6. Text requirement:
#    For each reported difference add a short, clear **human-readable description** (1–2 concise sentences) that states **what changed** and **where**.
#    * For **MODIFY/REPLACE**, include `before → after` when visible (e.g., OCR text/price).
#    * Keep descriptions neutral and standardized.

# </INSTRUCTION>

# <OUTPUT FORMAT>
# Always respond strictly in JSON with the following structure:

# {
# "same_group": true | false,
# "differences": [
# {
# "change_type": "ADD" | "REMOVE" | "MODIFY" | "REORDER" | "REPLACE",
# "description": "short human-readable summary",
# "before": "<string>",         // optional
# "after": "<string>"           // optional
# }
# ]
# }

# Rules:
# * Include `"differences"` only for the five allowed change types.
# * Omit optional fields if not applicable.
# * If no allowed differences are found, return `"differences": []` and `"same_group": true`.
# </OUTPUT FORMAT>

# <MAIN TASK>
# Compare the two onboarding videos, detect if they are in the same A/B test group (based only on the five allowed change types), and output the result in JSON.
# """

PROMPT_FOR_FIND_DIFFERENCE_V2 = """ 
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
"""



PROMPT_FOR_FIND_DIFFERENCE = """ 
<INSTRUCTION>
You are analyzing **two silent screen recordings** of the onboarding process of the same iOS application.
Each video was recorded by a different agent while going through the onboarding flow.

Your main goal:

1. Decide whether the two videos belong to the **same A/B test group** or to **different groups**.

2. Compare them screen-by-screen and capture only **meaningful differences** that fit exactly one of the following change types (STRICT ENUM):

   * **ADD** — an element/step/screen was added.
   * **REMOVE** — an element/step/screen was removed.
   * **MODIFY** — an attribute changed (copy/text, color, size, CTA text, price, trial\_days, order of bullets, etc.).
   * **REORDER** — the order of elements/steps changed.
   * **REPLACE** — one element was replaced by another (e.g., icon → image).

3. **Ignore everything else** (do NOT treat as A/B):
   timing/animation speed, skeleton loaders, network delays, device/screen-size responsive reflow without role change, status bar (battery/wifi/time), notifications, cursor movement, OS visual style differences, user choosing a different path when the same options are visible in both versions.

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
"same\_group": true | false,
"differences": \[
{
"change\_type": "ADD" | "REMOVE" | "MODIFY" | "REORDER" | "REPLACE",
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

\</OUTPUT FORMAT>

<MAIN TASK>
Compare the two onboarding videos, detect if they are in the same A/B test group (based only on the five allowed change types), and output the result in JSON.

<MOTIVATION>
Your accuracy is critical for detecting A/B tests. Be precise, exhaustive within the five types, and avoid assumptions outside of them.
"""




####
###"""
###<INSTRUCTION> 

###You are analyzing **two silent screen recordings** of the onboarding process of the same iOS application. Each video was recorded by a different agent while going through the onboarding flow.

###Your main goal: 1. Detect whether the two videos belong to the **same A/B test group** or to **different groups**. 2. To do this, compare them screen by screen and analyze all meaningful differences: - Screens, steps, texts, fonts, buttons, icons, checkmarks, layouts, images, animations, order of elements, etc. 3. Ignore irrelevant details: - battery/wifi indicators, notifications, time, cursor movement, minor glitches. Also, ignore small incosistent details. Remember, that A/B tests are always meaningful differences. 4. If the videos are identical in terms of functionality and flow → they belong to the **same group**. 5. If there are any functional differences → they belong to **different groups**. 6. Keep in mind that these 2 videos are made by a bot and it may go through onboarding differently - there may be delays or the bot may press different buttons - so watch and make sure you really see different AB groups. <OUTPUT FORMAT> Always respond strictly in JSON with the following structure: { "same_group": true | false, "differences": [ "difference_1", "difference_2", ... ] } - "same_group": true → if no meaningful differences are found. - "same_group": false → if at least one meaningful difference is found. - "differences" → list of all differences you identified (empty if none). </INSTRUCTION> <MAIN TASK> Compare the two onboarding videos, detect if they are in the same A/B test group, and output the result in JSON. </MAIN TASK> <MOTIVATION> Your accuracy is **critical** for detecting A/B tests. Be precise, exhaustive, and avoid assumptions. </MOTIVATION>
###"""


