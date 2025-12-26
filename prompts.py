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


# =============================================================================
# OLD JUDGE_PROMPT_TEMPLATE (All-or-nothing logic - commented out)
# =============================================================================
# JUDGE_PROMPT_TEMPLATE_OLD = """
# <ROLE>
#    You are an expert Video Quality Assurance Judge. 
#    Your ONLY task is to verify if the reported differences between two videos are REAL or HALLUCINATIONS.
# </ROLE>
#
# <INPUT>
#    Here are the claimed differences found by a previous analysis:
#    {differences}
# </INPUT>
#
# <TASK>
#    1. Watch the two attached videos carefully.
#    2. Check SPECIFICALLY for the differences listed above.
#    3. Determine if these differences actually exist in the video footage provided.
#       - Ignore minor timing differences or frame offsets.
#       - Ignore different start times (videos were trimmed randomly).
#       - Focus on CONTENT: visual elements, text, layout, flow.
# </TASK>
#
# <OUTPUT>
#    Reply strictly with JSON:
#    {
#    "reasoning": "Explanation and your thought process about the differences",
#    "verified": true | false,
#    }
# </OUTPUT>
# """

# =============================================================================
# NEW JUDGE_PROMPT_TEMPLATE (At-least-one logic)
# =============================================================================
JUDGE_PROMPT_TEMPLATE = """
<ROLE>
   You are an expert Video Quality Assurance Judge.
   Your task is to verify EACH reported difference individually and determine if AT LEAST ONE is REAL.
</ROLE>

<INPUT>
   Here are the claimed differences found by a previous analysis:
   {differences}
</INPUT>

<TASK>
   1. Watch the two attached videos carefully.
   2. Evaluate EACH claimed difference INDEPENDENTLY:
      - Is this specific difference REAL (actually visible in the videos)?
      - Or is it a HALLUCINATION (not actually present)?
   3. For each difference, mark it as verified (true) or not (false).
   
   IMPORTANT GUIDELINES:
   - Ignore minor timing differences or frame offsets.
   - Ignore different start times (videos were trimmed randomly).
   - Focus on CONTENT: visual elements, text, layout, flow.
   - A difference is REAL if the described content/element/screen actually differs between videos.
   - A difference is a HALLUCINATION if both videos show the same thing despite the claim.
</TASK>

<OUTPUT>
   Reply strictly with JSON:
   {{
      "per_difference_analysis": [
         {{
            "difference_index": 1,
            "description_summary": "Brief summary of what was claimed",
            "is_real": true | false,
            "explanation": "Why you determined this is real or hallucination"
         }}
      ],
      "reasoning": "Overall summary of your verification process",
      "verified_count": <number of differences that are REAL>,
      "total_count": <total number of differences evaluated>,
      "verified": true | false  // TRUE if verified_count >= 1, FALSE if verified_count == 0
   }}
</OUTPUT>

<DECISION RULE>
   verified = TRUE if AT LEAST ONE difference is real (verified_count >= 1)
   verified = FALSE only if ALL differences are hallucinations (verified_count == 0)
</DECISION RULE>
"""

# =============================================================================
# OLD JUDGE_FOCUSED_PROMPT_TEMPLATE (commented out for reference)
# =============================================================================
# JUDGE_FOCUSED_PROMPT_TEMPLATE_OLD = """
# <ROLE>
#    You are an expert Video Quality Assurance Judge.
#    Your ONLY task is to verify if the reported difference between two videos is REAL or a HALLUCINATION.
# </ROLE>
#
# <CONTEXT>
#    You are viewing SHORT CLIPS extracted specifically around the timestamp where a difference was reported.
#    These clips include a few seconds of buffer before and after the reported event.
#    
#    NOTE: The timing in these clips is relative to the clip start, NOT the original video. 
#    You must look for the SPECIFIC content difference described, regardless of exact second it appears in this clip.
# </CONTEXT>
#
# <INPUT>
#    Here is the claimed difference found by a previous analysis:
#    {differences}
# </INPUT>
#
# <TASK>
#    1. Watch the two attached short clips carefully.
#    2. Check SPECIFICALLY for the difference described above.
#    3. Determine if this difference actually exists in the footage provided.
#       - Ignore start/end cutoffs (these are trimmed clips).
#       - Focus on CONTENT: Is the described element/text/step present in one and absent/different in the other?
# </TASK>
#
# <OUTPUT>
#    Reply strictly with JSON:
#    {
#    "reasoning": "Detailed explanation of what you see in the clips relative to the claimed difference",
#    "verified": true | false
#    }
# </OUTPUT>
# """

# =============================================================================
# NEW JUDGE_FOCUSED_PROMPT_TEMPLATE (Single difference - clearer logic)
# =============================================================================
JUDGE_FOCUSED_PROMPT_TEMPLATE = """
<ROLE>
   You are an expert Video Quality Assurance Judge.
   Your task is to verify if the reported difference is REAL or a HALLUCINATION.
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
   
   GUIDELINES:
   - Ignore start/end cutoffs (these are trimmed clips).
   - Ignore exact timing - focus on whether the CONTENT difference exists.
   - A difference is REAL if the described element/text/screen is genuinely different between videos.
   - A difference is a HALLUCINATION if both clips show essentially the same thing.
</TASK>

<OUTPUT>
   Reply strictly with JSON:
   {{
      "reasoning": "Detailed explanation of what you see in the clips relative to the claimed difference",
      "is_real": true | false,
      "verified": true | false  // Same as is_real for single difference
   }}
</OUTPUT>
"""

# =============================================================================
# OLD JUDGE_SIDE_BY_SIDE_PROMPT_TEMPLATE (commented out for reference)
# =============================================================================
# JUDGE_SIDE_BY_SIDE_PROMPT_TEMPLATE_OLD = """
# <ROLE>
#    You are an expert Video Quality Assurance Judge.
#    Your ONLY task is to verify if the reported difference is REAL or a HALLUCINATION.
# </ROLE>
#
# <CONTEXT>
#    You are viewing a SINGLE VIDEO containing two clips stacked side-by-side.
#    - LEFT SIDE: Video 1 (Original/First)
#    - RIGHT SIDE: Video 2 (Second)
#    
#    These clips are extracted around the reported timestamp.
#    IGNORE TIMING differences completely. Focus ONLY on the actual content difference.
# </CONTEXT>
#
# <INPUT>
#    Here is the claimed difference found by a previous analysis:
#    {differences}
# </INPUT>
#
# <TASK>
#    1. Watch the side-by-side video carefully.
#    2. Compare the LEFT side vs the RIGHT side.
#    3. Check SPECIFICALLY for the difference described above.
#    4. Determine if this difference actually exists in the content.
#       - Ignore different start/end cutoffs.
#       - Ignore animation sync (unless the difference IS the animation type).
#       - Focus on CONTENT: visual elements, text, layout, flow.
# </TASK>
#
# <OUTPUT>
#    Reply strictly with JSON:
#    {
#    "reasoning": "Detailed explanation of what you see comparing Left vs Right",
#    "verified": true | false
#    }
# </OUTPUT>
# """

# =============================================================================
# NEW JUDGE_SIDE_BY_SIDE_PROMPT_TEMPLATE (Single difference - clearer logic)
# =============================================================================
JUDGE_SIDE_BY_SIDE_PROMPT_TEMPLATE = """
<ROLE>
   You are an expert Video Quality Assurance Judge.
   Your task is to verify if the reported difference is REAL or a HALLUCINATION.
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
   
   GUIDELINES:
   - Ignore different start/end cutoffs.
   - Ignore animation sync (unless the difference IS about the animation type itself).
   - Focus on CONTENT: visual elements, text, layout, flow.
   - A difference is REAL if LEFT and RIGHT genuinely show different content as described.
   - A difference is a HALLUCINATION if LEFT and RIGHT show essentially the same thing.
</TASK>

<OUTPUT>
   Reply strictly with JSON:
   {{
      "reasoning": "Detailed explanation of what you see comparing Left vs Right",
      "is_real": true | false,
      "verified": true | false  // Same as is_real for single difference
   }}
</OUTPUT>
"""

# AB_TEST_CLASSIFIER_PROMPT = """
# <ROLE>
#    You are a Senior Product Manager analyzing confirmed differences between two app recordings.
#    Your ONLY task is to determine if the confirmed difference represents an ACTUAL A/B TEST.
# </ROLE>

# <CONTEXT>
#    A QA Judge has already verified that a difference exists between two video recordings of the same app.
#    Now you must determine if this difference is an intentional A/B test variation or something else.
# </CONTEXT>

# <INPUT>
#    ORIGINAL DIFFERENCE DETECTED:
#    {differences}
   
#    QA JUDGE'S VERIFICATION:
#    {judge_reasoning}
# </INPUT>

# <TASK>
#    Analyze the confirmed difference and classify it:
   
#    ✅ IS AN A/B TEST - Intentional product variations:
#       - Different UI layouts, button placements, or visual designs
#       - Different button styles, colors, or sizes
#       - Different text/copy variations for the same functionality
#       - Different feature implementations or user flows
#       - Presence/absence of UI elements or features
#       - Different pricing displays or subscription options
#       - Different onboarding flows or tutorial content
   
#    ❌ NOT AN A/B TEST - Non-product differences:
#       - Different user accounts (emails, usernames, profile photos)
#       - Different user-generated content (posts, messages, history)
#       - Different timestamps, dates, or time-sensitive data
#       - Different notification counts or badges
#       - Different dynamic content (feed items, recommendations)
#       - Random system variations (loading states, cached data)
#       - Network-dependent content differences
#       - Semantically identical content with only personal data changed
   
#    KEY PRINCIPLE: If the only difference is USER DATA (account info, personal content) 
#    while the UI/UX and product functionality are IDENTICAL, it is NOT an A/B test.
# </TASK>

# <OUTPUT>
#    Reply strictly with JSON:
#    {
#    "is_ab_test": true | false,
#    "reasoning": "Explain your classification. Why is this or isn't this an A/B test?"
#    }
# </OUTPUT>
# """


AB_TEST_CLASSIFIER_PROMPT = """
<ROLE>
You are a Senior Mobile Product Analyst who specializes in A/B test detection for iOS applications.
Your expertise: distinguishing INTENTIONAL product experiments from SYSTEM-LEVEL or INCIDENTAL differences.
</ROLE>

<CONTEXT>
A QA system has verified that a visual difference exists between two recordings of the same iOS app.
Your job: Determine if this difference is an INTENTIONAL A/B TEST designed by the app's product team.
</CONTEXT>

<CORE PRINCIPLE>
A/B tests are PRODUCT DECISIONS. The app developers deliberately created two different experiences 
to measure which performs better. If the app team did NOT control this difference, it's NOT an A/B test.
</CORE PRINCIPLE>

<DECISION TREE>
Follow this decision tree IN ORDER. Stop at the first YES.

1. Is this an iOS SYSTEM UI element?
   → SKStoreReviewController ("Rate this app" prompt)
   → Native permission dialogs (photos, camera, notifications, ATT, location)
   → iOS keyboard variations
   → System alerts, share sheets, or action sheets with apple styling
   → Status bar elements (battery, time, signal, carrier)
   
   If YES → NOT an A/B test (Apple/iOS controls this, not the app)

2. Is this TIMING or SYNCHRONIZATION of the same element?
   → Same dialog appearing at different seconds
   → Same animation at different phase
   → Same loading state at different moment
   
   If YES → NOT an A/B test (Recording timing variance, not product difference)

3. Is this USER-SPECIFIC or SERVER-STATE data?
   → User account info (email, name, avatar)
   → Notification badges or counts
   → Dynamic content from backend (feed items, recommendations)
   → Cached data differences
   
   If YES → NOT an A/B test (External data, not product design)

4. Is this APP-DESIGNED UI/UX that differs structurally?
   → Different screens in the flow (added/removed/reordered)
   → Different visual design (colors, layouts, button styles)
   → Different text copy or CTA wording
   → Different pricing, offers, or trial periods
   → Different features shown or hidden
   → Custom app-designed screens that explain permissions (not the iOS dialog itself)
   
   If YES → IS an A/B test (Product team designed this variation)

5. None of the above matched clearly?
   → Ask: "Could a product manager have DECIDED to test this?"
   → If the variation requires CODE CHANGES to implement → likely A/B test
   → If the variation happens WITHOUT code changes (system/timing/user state) → NOT A/B test
</DECISION TREE>

<EXAMPLES>
EXAMPLE 1:
Difference: "An iOS App Store rating prompt asking 'Enjoying App? Tap a star to rate' appears in Video 1 but not Video 2"
Analysis: This is SKStoreReviewController - an iOS system UI. Apple's algorithm decides when to show it, not the app.
Decision: NOT an A/B test

EXAMPLE 2:
Difference: "The 'Why we need your photos' explanation screen has different text copy in Video 1 vs Video 2"
Analysis: This is a custom app-designed screen (not the iOS permission dialog). Different copy = product decision.
Decision: IS an A/B test

EXAMPLE 3:
Difference: "The iOS photo library permission dialog appears at 0:30 in Video 1 but 0:45 in Video 2"
Analysis: Same system dialog, different timing. This is recording/user-pace variance, not a product decision.
Decision: NOT an A/B test

EXAMPLE 4:
Difference: "Video 1 shows a privacy consent screen before onboarding, Video 2 skips directly to features"
Analysis: Presence/absence of an app-designed screen = deliberate flow variation = product decision.
Decision: IS an A/B test

EXAMPLE 5:
Difference: "The notification badge shows '3' in Video 1 and '7' in Video 2"
Analysis: Badge count is server/user state, not a designed UI variation.
Decision: NOT an A/B test

EXAMPLE 6:
Difference: "Video 1 shows a 'Rate Us' custom popup with app branding, Video 2 doesn't show it"
Analysis: This is a CUSTOM popup (app-designed), not the iOS SKStoreReviewController. Presence of custom UI = product decision.
Decision: IS an A/B test
</EXAMPLES>

<INPUT>
VERIFIED DIFFERENCE:
{differences}

VERIFICATION CONTEXT:
{judge_reasoning}
</INPUT>

<TASK>
1. First, identify which decision tree branch this difference falls into
2. Explain your reasoning step by step
3. Make your classification

Output strictly as JSON:
{{
  "decision_path": "Which decision tree step (1-5) determined your answer",
  "reasoning": "Your step-by-step analysis",
  "is_ab_test": true | false
}}
</TASK>
"""