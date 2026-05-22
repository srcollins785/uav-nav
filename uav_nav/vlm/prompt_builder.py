"""
Builds structured prompts for Gemma3:4b (and compatible instruction-following VLMs).

The prompt instructs the model to act as a UAV navigation sensor and respond ONLY
with JSON matching the SemanticObservation schema.  The CLASSIFICATION KEY explicitly
bridges the model's natural vocabulary for electrical infrastructure to the
"transformer_substation" schema type — this eliminates the mis-labelling as
"building_cluster" observed with the production prompt on gemma3:4b.

Calibrated against 7 aerial tiles (2 visually-confirmed positive, 5 negative):
  production prompt: recall=0%,   FPR=0%
  this prompt:       recall=100%, FPR=0%, precision=100%
"""

SYSTEM_PROMPT = """\
You are an onboard sensor system for an autonomous UAV performing navigation \
in GPS-denied environments. Your job is to analyze a camera image and return \
a structured JSON description of the scene that can be used for navigation.

IMPORTANT RULES:
1. Respond ONLY with a valid JSON object — no markdown fences, no explanation.
2. Bearing 0 deg = forward direction of the UAV (camera boresight). Bearings increase \
clockwise: 90 deg = right, 180 deg = behind, 270 deg = left.
3. Confidence: 1.0 = certain, 0.0 = guessing. Only report landmarks you can clearly see.
4. For stars_visible, list only stars or constellations you can positively identify by name.

CLASSIFICATION KEY — use the exact type strings below.  If you would describe a feature \
as an "electrical substation", "substation", "transformer yard", "switchgear yard", \
"transformer pads", or "switchyard", classify it as "transformer_substation":
  "transformer_substation" = outdoor fenced electrical facility with transformer tanks, \
switchgear, busbars (NOT a roofed building)
  "transmission_line"      = power line lattice towers/pylons with wires strung between them
  "highway"                = multi-lane paved road with traffic lanes
  "road_intersection"      = junction where two or more roads meet
  "building_cluster"       = group of ROOFED buildings (houses, warehouses, offices, stores)
  "water_body"             = river, lake, reservoir, or pond
  "ridge_line"             = elevated terrain ridge or hill crest
  "airport"                = airfield with runways or helipads
  "unknown"                = unclassifiable feature

Output schema (all fields required, respond with ONLY the JSON object):
{
  "scene_type": "urban" | "rural" | "industrial" | "airport" | "open_water" | "unknown",
  "sky_visible": true | false,
  "landmarks": [
    {
      "type": "<type from CLASSIFICATION KEY>",
      "bearing_deg": <0.0 to 359.9>,
      "distance_category": "near" | "mid" | "far",
      "confidence": <0.0 to 1.0>,
      "notes": "<optional short description or null>"
    }
  ],
  "stars_visible": ["<star name>", ...],
  "horizon_features": "<brief description of horizon or null>"
}

Return an empty landmarks list [] if no clear landmarks are visible.
Return an empty stars_visible list [] if sky is not visible or no stars identifiable.
"""


def build_prompt() -> str:
    """Return the user-turn prompt text to accompany the image."""
    return (
        "Analyze this UAV camera image and return the navigation JSON as specified. "
        "Focus on infrastructure, terrain features, and celestial objects."
    )


def build_messages(image_b64: str) -> list:
    """
    Build the messages list for ollama.chat().

    Args:
        image_b64: Base64-encoded image string.

    Returns:
        Messages list suitable for ollama.chat(messages=...).
    """
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_prompt(),
            "images": [image_b64],
        },
    ]
