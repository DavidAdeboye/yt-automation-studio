---
title: YT Automation Studio
emoji: "✂️"
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# YT Automation Studio

A constrained, free-host-friendly video clipping MVP. It uses YouTube captions to identify promising moments and downloads only short selected ranges rather than the full VOD.

## Hugging Face setup

Create a Docker Space and upload this repository. In the Space settings, add this secret:

- `GROQ_API_KEY`: a Groq API key used for highlight selection and clip transcription.

The Space runs one video job at a time. Generated files live in temporary storage and may disappear whenever the Space restarts.

## Current clipping limits

- Up to 3 clips per job
- 20–60 seconds per clip
- Source segments capped at 720p
- No full-VOD fallback
- Manual timestamps available when English captions are missing

Only process videos you have permission to use, and follow YouTube's terms and applicable copyright rules.
