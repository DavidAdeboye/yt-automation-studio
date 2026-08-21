"""
Shared utilities — all API-based, no GPU/local models required.

- LLM scripting        -> Groq (free tier, OpenAI-compatible endpoint)
- Voice narration       -> Edge-TTS (free, cloud-based)
- Captions/transcription -> Groq's Whisper API (free tier, hosted)
- Scene images          -> Pollinations.ai (free, no API key needed)

Env vars needed: GROQ_API_KEY
"""

import os
import subprocess
import asyncio
import urllib.parse
import requests


AGENTROUTER_API_KEY = os.environ.get("AGENTROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "")).strip()
AGENTROUTER_BASE_URL = os.environ.get("AGENTROUTER_BASE_URL", "https://agentrouter.org/v1").rstrip("/")
AGENTROUTER_MODEL = os.environ.get("AGENTROUTER_MODEL", os.environ.get("OPENAI_MODEL", "gpt-5.6")).strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "").strip()
_SELECTED_GROQ_MODEL = None


def _groq_error(response):
    try:
        detail = response.json().get("error", {}).get("message")
    except ValueError:
        detail = response.text.strip()
    return detail or f"HTTP {response.status_code}"


def _select_groq_model() -> str:
    """Select an available chat model, while allowing an explicit override."""
    global _SELECTED_GROQ_MODEL
    if GROQ_MODEL:
        return GROQ_MODEL
    if _SELECTED_GROQ_MODEL:
        return _SELECTED_GROQ_MODEL

    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"Groq model lookup failed ({response.status_code}): {_groq_error(response)}")
    model_ids = [item["id"] for item in response.json().get("data", [])]
    excluded = ("whisper", "guard", "tts", "speech", "audio", "moderation")
    chat_models = [
        model_id for model_id in model_ids
        if not any(word in model_id.lower() for word in excluded)
    ]
    preferred = (
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    )
    _SELECTED_GROQ_MODEL = next(
        (model_id for model_id in preferred if model_id in chat_models),
        chat_models[0] if chat_models else None,
    )
    if not _SELECTED_GROQ_MODEL:
        raise RuntimeError(
            "Your Groq account has no available chat model. Add GROQ_MODEL in app secrets "
            "after choosing a model listed in the Groq console."
        )
    return _SELECTED_GROQ_MODEL


# ---------- LLM scripting ----------

def call_llm(prompt: str, model: str = None) -> str:
    if AGENTROUTER_API_KEY:
        chosen_model = model or AGENTROUTER_MODEL
        response = requests.post(
            f"{AGENTROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {AGENTROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": chosen_model, "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(f"AgentRouter request failed ({response.status_code}): {_groq_error(response)}")
        return response.json()["choices"][0]["message"]["content"].strip()
    if not GROQ_API_KEY:
        raise RuntimeError("Add AGENTROUTER_API_KEY or GROQ_API_KEY to the app secrets.")
    model = model or _select_groq_model()
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
    )
    if not response.ok:
        raise RuntimeError(f"Groq chat request failed ({response.status_code}): {_groq_error(response)}")
    return response.json()["choices"][0]["message"]["content"].strip()


# ---------- Narration ----------

async def _synthesize(text: str, output_path: str, voice: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def narrate(text: str, output_path: str, voice: str = "en-US-GuyNeural") -> str:
    asyncio.run(_synthesize(text, output_path, voice))
    return output_path


def get_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrapper=1:nokey=1", audio_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


# ---------- Captions (via Groq's hosted Whisper API — no local model) ----------

def transcribe_audio(audio_or_video_path: str) -> str:
    """Returns an SRT-formatted transcript using Groq's hosted whisper-large-v3."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured in the app secrets.")
    with open(audio_or_video_path, "rb") as f:
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": f},
            data={"model": "whisper-large-v3", "response_format": "srt"},
        )
    if not response.ok:
        raise RuntimeError(f"Groq transcription failed ({response.status_code}): {_groq_error(response)}")
    return response.text


def burn_captions(video_path: str, output_path: str):
    srt_text = transcribe_audio(video_path)
    srt_path = os.path.splitext(video_path)[0] + ".srt"
    with open(srt_path, "w") as f:
        f.write(srt_text)

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"subtitles={srt_path}",
        output_path,
    ], check=True)
    return output_path


# ---------- Scene images (Pollinations.ai — free, no key, pure HTTP GET) ----------

def generate_scene_image(prompt: str, output_path: str, width: int = 1080, height: int = 1920):
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path


# ---------- Shared ffmpeg helpers ----------

def apply_zoompan(image_path: str, duration: float, output_path: str, fps: int = 30):
    frames = int(duration * fps)
    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", image_path,
        "-vf", f"zoompan=z='min(zoom+0.0015,1.3)':d={frames}:s=1080x1920:fps={fps}",
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path,
    ], check=True)
    return output_path


def combine_video_and_audio(video_path: str, audio_path: str, output_path: str):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path,
    ], check=True)
    return output_path


def concatenate_clips(clip_paths: list, output_path: str):
    list_file = "concat_list.txt"
    with open(list_file, "w") as f:
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_path,
    ], check=True)
    os.remove(list_file)
    return output_path
