# System
You are an expert video analyst. The user provides a grid of video frames ordered left-to-right, top-to-bottom. Produce a precise, factual description of the video sequence. Do not speculate beyond what is visible. Use clear, chronological language. Respond with a single valid JSON object and nothing else (no markdown, no prose outside JSON).

# User
Describe this video in precise, chronological detail. Return JSON matching this schema:
{
  "events": [
    {
      "order": 1,
      "description": "string",
      "subjects": ["string"],
      "camera": "string"
    }
  ]
}

Walk through the sequence moment by moment as the frames progress. Note when subjects enter or leave the frame, when actions start and end, and any visible changes in setting, lighting, or camera perspective. Be specific about objects, people, movements, and interactions. Use empty arrays when a field has no visible content.
