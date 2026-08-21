"""
YT Automation Studio — Streamlit app, queued version.

Same four niches as before (Clipping, Ranking, Reddit Stories, History/What-If),
but pipeline runs are submitted to a background job queue (job_queue.py) instead
of blocking the browser tab. A "Job Queue" panel in the sidebar shows status for
everything submitted, and you refresh manually to check progress.

Run with: streamlit run app.py
Env var needed: GROQ_API_KEY
"""

import streamlit as st
import os
import json
import tempfile
import subprocess
import random
import re
from urllib.parse import parse_qs, urlparse

from utils import (
    call_llm, narrate, get_audio_duration, burn_captions,
    generate_scene_image, apply_zoompan, combine_video_and_audio,
    concatenate_clips,
)
import job_queue

st.set_page_config(page_title="YT Automation Studio", layout="wide")


def workdir():
    return tempfile.mkdtemp(prefix="ytauto_")


def run_command(command: list[str], description: str):
    """Run a CLI command and preserve its useful diagnostic on failure."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        diagnostic = (result.stderr or result.stdout or "No diagnostic output.").strip()
        # Keep queue errors readable while retaining the final, usually decisive lines.
        diagnostic = "\n".join(diagnostic.splitlines()[-20:])
        raise RuntimeError(f"{description} failed:\n{diagnostic}")
    return result


def youtube_video_id(url: str) -> str:
    """Return a compact label for common YouTube URL forms."""
    parsed = urlparse(url)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/") or url
    if parsed.path == "/watch":
        return parse_qs(parsed.query).get("v", [url])[0]
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] in {"live", "shorts", "embed"}:
        return parts[1]
    return url


# ==================================================================
# Pipeline functions — plain Python, no st.* calls (run on a worker thread)
# ==================================================================

def timestamp_seconds(value: str) -> float:
    """Convert SS, MM:SS or HH:MM:SS into seconds."""
    parts = value.strip().split(":")
    if not parts or len(parts) > 3:
        raise ValueError(f"Invalid timestamp: {value}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value}") from exc
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def parse_manual_timestamps(raw: str, clip_duration: int, limit: int) -> list[dict]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Enter at least one timestamp, for example 1:23:45.")
    return [
        {"start": max(0, timestamp_seconds(value) - 5), "reason": "Manual timestamp"}
        for value in values[:limit]
    ]


def vtt_to_transcript(vtt_text: str) -> str:
    """Create a compact timestamped transcript from a YouTube WebVTT file."""
    entries = []
    last_text = None
    blocks = re.split(r"\n\s*\n", vtt_text.replace("\r", ""))
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        start = lines[timing_index].split("-->", 1)[0].strip().split(".", 1)[0]
        text = " ".join(lines[timing_index + 1:])
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text != last_text:
            entries.append(f"[{start}] {text}")
            last_text = text
    return "\n".join(entries)


def find_transcript_highlights(stream_url: str, num_clips: int, wd: str) -> list[dict]:
    subtitle_template = os.path.join(wd, "source.%(ext)s")
    run_command([
        "yt-dlp", "--no-playlist", "--no-progress", "--skip-download",
        "--write-subs", "--write-auto-subs", "--sub-langs", "en,en-*",
        "--sub-format", "vtt", "-o", subtitle_template, stream_url,
    ], "Caption lookup")
    subtitle_paths = [
        os.path.join(wd, name) for name in os.listdir(wd) if name.endswith(".vtt")
    ]
    if not subtitle_paths:
        raise RuntimeError(
            "No English captions were found. Use Manual timestamps; the app will not "
            "download the full video as a fallback."
        )
    with open(subtitle_paths[0], "r", encoding="utf-8") as f:
        transcript = vtt_to_transcript(f.read())
    if not transcript:
        raise RuntimeError("Captions were found but could not be read. Use Manual timestamps.")

    # Bound prompt size while retaining coverage across a long VOD.
    # Groq's free tier has a small tokens-per-minute limit. Keep the prompt
    # around 6k tokens so the model can still return the requested JSON.
    max_chars = 22000
    if len(transcript) > max_chars:
        lines = transcript.splitlines()
        stride = max(1, len(transcript) // max_chars + 1)
        transcript = "\n".join(lines[::stride])
    raw = call_llm(
        f"Choose exactly {num_clips} distinct, highly clip-worthy moments from this "
        "timestamped livestream transcript. Favor surprising, funny, emotional, tense, "
        "or self-contained moments. Return ONLY a JSON array with numeric start seconds "
        "and a short reason, like [{'{'}\"start\": 123.0, \"reason\": \"...\"{'}'}]. "
        "Place start 5 seconds before the key line when possible.\n\n" + transcript
    )
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise RuntimeError(f"Highlight selection returned no JSON array. Model said: {raw[:500]}")
    try:
        highlights = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Highlight selection returned invalid JSON ({exc}). Model said: {raw[:500]}"
        ) from exc
    clean = []
    for item in highlights[:num_clips]:
        start = max(0.0, float(item["start"]))
        clean.append({"start": start, "reason": str(item.get("reason", "Highlight"))})
    if not clean:
        raise RuntimeError("No highlights were selected.")
    return clean


def pipeline_clipping(stream_url: str, num_clips: int, clip_duration: int,
                      selection_mode: str, manual_timestamps: str, wd: str) -> list:
    if selection_mode == "Manual timestamps":
        highlights = parse_manual_timestamps(manual_timestamps, clip_duration, num_clips)
    else:
        highlights = find_transcript_highlights(stream_url, num_clips, wd)

    results = []
    for i, highlight in enumerate(highlights):
        start = highlight["start"]
        end = start + clip_duration
        raw_template = os.path.join(wd, f"clip_{i}_source.%(ext)s")
        run_command([
            "yt-dlp", "--no-playlist", "--no-progress",
            "--download-sections", f"*{start:.3f}-{end:.3f}", "--force-keyframes-at-cuts",
            "-f", "bv*[height<=720]+ba/b[height<=720]/b",
            "--merge-output-format", "mp4", "-o", raw_template, stream_url,
        ], f"Clip {i + 1} download")
        candidates = [
            os.path.join(wd, name) for name in os.listdir(wd)
            if name.startswith(f"clip_{i}_source.") and not name.endswith(".part")
        ]
        if not candidates:
            raise RuntimeError(f"Clip {i + 1} download produced no video file.")
        raw = candidates[0]
        vertical = os.path.join(wd, f"clip_{i}_vertical.mp4")
        final = os.path.join(wd, f"clip_{i}_final.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", raw,
            "-vf", "crop=ih*9/16:ih,scale=1080:1920", "-c:a", "copy", vertical,
        ], check=True)
        burn_captions(vertical, final)
        for intermediate in (raw, vertical):
            if os.path.exists(intermediate):
                os.remove(intermediate)
        results.append(final)
    return results


def pipeline_ranking(niche_context: str, items: list, wd: str) -> str:
    """items: list of dicts with rank, title, stat, footage_path (already saved to disk)."""
    segment_paths = []
    for item in items:
        script = call_llm(
            f"Write a punchy 2-3 sentence countdown commentary line for a "
            f"{niche_context} ranking video. Rank #{item['rank']}: {item['title']}. "
            f"Key detail: {item['stat']}. Energetic sports countdown narrator tone, no filler."
        )
        narration_path = os.path.join(wd, f"narration_{item['rank']}.mp3")
        narrate(script, narration_path)

        segment_path = os.path.join(wd, f"segment_{item['rank']}.mp4")
        combine_video_and_audio(item["footage_path"], narration_path, segment_path)
        segment_paths.append(segment_path)

    raw = os.path.join(wd, "countdown_raw.mp4")
    concatenate_clips(segment_paths, raw)
    final = os.path.join(wd, "countdown_final.mp4")
    burn_captions(raw, final)
    return final


def pipeline_reddit(story_text: str, footage_paths: list, wd: str) -> str:
    script = call_llm(
        "Rewrite this Reddit-style post as a spoken narration script for a short-form "
        "video. Start with a punchy one-line hook. Keep the actual story/events intact, "
        "just make it flow naturally read aloud. No headers, just narration text.\n\n"
        + story_text
    )
    narration_path = os.path.join(wd, "narration.mp3")
    narrate(script, narration_path, voice="en-US-AriaNeural")
    duration = get_audio_duration(narration_path)

    chosen = random.choice(footage_paths)
    trimmed = os.path.join(wd, "footage_trimmed.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", chosen,
        "-t", str(duration), "-an", "-c:v", "libx264", trimmed,
    ], check=True)

    combined = os.path.join(wd, "combined.mp4")
    combine_video_and_audio(trimmed, narration_path, combined)
    final = os.path.join(wd, "reddit_final.mp4")
    burn_captions(combined, final)
    return final


def pipeline_history(topic: str, num_scenes: int, wd: str) -> str:
    raw_json = call_llm(
        f"Write a short-form narrated video script about: \"{topic}\". "
        f"Split it into exactly {num_scenes} scenes. Start with a hook scene. "
        "Return ONLY a JSON array, no other text, format: "
        '[{"narration": "...", "image_prompt": "..."}, ...]'
    )
    scenes = json.loads(raw_json)

    scene_paths = []
    for i, scene in enumerate(scenes):
        narration_path = os.path.join(wd, f"scene_{i}_narration.mp3")
        narrate(scene["narration"], narration_path, voice="en-US-ChristopherNeural")
        duration = get_audio_duration(narration_path)

        image_path = os.path.join(wd, f"scene_{i}_image.png")
        generate_scene_image(scene["image_prompt"], image_path)

        zoompan_path = os.path.join(wd, f"scene_{i}_zoompan.mp4")
        apply_zoompan(image_path, duration, zoompan_path)

        scene_final = os.path.join(wd, f"scene_{i}_final.mp4")
        combine_video_and_audio(zoompan_path, narration_path, scene_final)
        scene_paths.append(scene_final)

    raw = os.path.join(wd, "history_raw.mp4")
    concatenate_clips(scene_paths, raw)
    final = os.path.join(wd, "history_final.mp4")
    burn_captions(raw, final)
    return final


# ==================================================================
# Sidebar: Job Queue panel
# ==================================================================

if "job_ids" not in st.session_state:
    st.session_state["job_ids"] = []

with st.sidebar:
    st.header("Job Queue")
    if st.button("Refresh status"):
        st.rerun()

    if not st.session_state["job_ids"]:
        st.caption("No jobs submitted yet.")
    else:
        for job_id in reversed(st.session_state["job_ids"]):
            info = job_queue.get_job_info(job_id)
            status = job_queue.get_status(job_id)
            icon = {"queued": "⏳", "running": "⚙️", "done": "✅", "error": "❌"}.get(status, "?")
            st.write(f"{icon} **{info.get('label', job_id)}** — {status}")
            if status == "error":
                st.caption(str(job_queue.get_error(job_id)))


# ==================================================================
# Main tabs
# ==================================================================

st.title("YT Automation Studio")

tab_clip, tab_rank, tab_reddit, tab_history = st.tabs(
    ["Clipping", "Ranking", "Reddit Stories", "History / What-If"]
)


def show_result_if_ready(job_id_key: str, filename: str):
    """Shared helper: if the job tied to this session key is done, show video + download."""
    job_id = st.session_state.get(job_id_key)
    if not job_id:
        return
    status = job_queue.get_status(job_id)
    if status == "done":
        result = job_queue.get_result(job_id)
        if isinstance(result, list):
            for path in result:
                st.video(path)
                with open(path, "rb") as f:
                    st.download_button(f"Download {os.path.basename(path)}", f,
                                        file_name=os.path.basename(path), key=path)
        elif result:
            st.video(result)
            with open(result, "rb") as f:
                st.download_button("Download video", f, file_name=filename, key=job_id_key)
    elif status == "error":
        st.error(f"Job failed: {job_queue.get_error(job_id)}")
    elif status in ("queued", "running"):
        st.info(f"Job is {status} — check the sidebar, click Refresh status when ready.")


# ---------------- Clipping ----------------

with tab_clip:
    st.subheader("Create clips without downloading the full VOD")
    st.caption("The app fetches captions first, then downloads only the selected short ranges (max 720p).")
    stream_url = st.text_input("Stream or VOD URL", key="clip_url")
    selection_mode = st.radio(
        "How should moments be selected?",
        ["Automatic from captions", "Manual timestamps"],
        horizontal=True,
        key="clip_selection_mode",
    )
    manual_timestamps = ""
    if selection_mode == "Manual timestamps":
        manual_timestamps = st.text_input(
            "Timestamps, separated by commas",
            placeholder="12:34, 1:23:45",
            key="clip_timestamps",
        )
    num_clips = st.slider("Max clips to generate", 1, 3, 1, key="clip_count")
    clip_duration = st.slider("Seconds per clip", 20, 60, 45, step=5, key="clip_duration")
    st.info(
        f"Hard limit: up to {num_clips} segment(s) × {clip_duration} seconds. "
        "The full VOD is never downloaded."
    )

    if st.button("Generate clips", key="clip_run"):
        if not stream_url:
            st.warning("Paste a link first.")
        elif selection_mode == "Manual timestamps" and not manual_timestamps.strip():
            st.warning("Enter at least one timestamp.")
        else:
            wd = workdir()
            job_id = job_queue.submit_job("clipping", f"Clipping: {youtube_video_id(stream_url)}",
                                           pipeline_clipping, stream_url, num_clips, clip_duration,
                                           selection_mode, manual_timestamps, wd)
            st.session_state["job_ids"].append(job_id)
            st.session_state["clip_job_id"] = job_id
            st.success("Job submitted — check the sidebar for status.")

    show_result_if_ready("clip_job_id", "clips.mp4")


# ---------------- Ranking ----------------

with tab_rank:
    st.subheader("Build a countdown/ranking video")
    niche_context = st.text_input("Niche context (e.g. 'football goals')", key="rank_niche")
    num_items = st.slider("Number of ranked items", 3, 15, 10, key="rank_count")

    st.write("Upload footage clips and give each a title + stat, in countdown order (worst to best):")
    raw_items = []
    for i in range(num_items):
        rank_num = num_items - i
        cols = st.columns([1, 2, 2, 3])
        cols[0].write(f"#{rank_num}")
        title = cols[1].text_input("Title", key=f"rank_title_{i}")
        stat = cols[2].text_input("Stat/fact", key=f"rank_stat_{i}")
        footage = cols[3].file_uploader("Footage clip", type=["mp4"], key=f"rank_footage_{i}")
        raw_items.append({"rank": rank_num, "title": title, "stat": stat, "footage": footage})

    if st.button("Generate ranking video", key="rank_run"):
        wd = workdir()
        items = []
        for item in raw_items:
            if not item["footage"]:
                continue
            footage_path = os.path.join(wd, f"footage_{item['rank']}.mp4")
            with open(footage_path, "wb") as f:
                f.write(item["footage"].read())
            items.append({**item, "footage_path": footage_path})

        if not items:
            st.warning("Upload at least one footage clip.")
        else:
            job_id = job_queue.submit_job("ranking", f"Ranking: {niche_context}",
                                           pipeline_ranking, niche_context, items, wd)
            st.session_state["job_ids"].append(job_id)
            st.session_state["rank_job_id"] = job_id
            st.success("Job submitted — check the sidebar for status.")

    show_result_if_ready("rank_job_id", "countdown_final.mp4")


# ---------------- Reddit Stories ----------------

with tab_reddit:
    st.subheader("Paste a story, get a narrated short")
    story_text = st.text_area("Paste the post text (title + body)", height=200, key="reddit_text")
    footage_lib = st.file_uploader(
        "Upload one or more background footage clips (a random one gets picked)",
        type=["mp4"], accept_multiple_files=True, key="reddit_footage",
    )

    if st.button("Generate story video", key="reddit_run"):
        if not story_text or not footage_lib:
            st.warning("Paste a story and upload at least one footage clip.")
        else:
            wd = workdir()
            footage_paths = []
            for i, f in enumerate(footage_lib):
                path = os.path.join(wd, f"footage_{i}.mp4")
                with open(path, "wb") as out:
                    out.write(f.read())
                footage_paths.append(path)

            job_id = job_queue.submit_job("reddit", f"Reddit story: {story_text[:30]}",
                                           pipeline_reddit, story_text, footage_paths, wd)
            st.session_state["job_ids"].append(job_id)
            st.session_state["reddit_job_id"] = job_id
            st.success("Job submitted — check the sidebar for status.")

    show_result_if_ready("reddit_job_id", "reddit_final.mp4")


# ---------------- History / What-If ----------------

with tab_history:
    st.subheader("Give it a topic, get a narrated explainer")
    topic = st.text_input("Topic (e.g. 'What if Rome never fell?')", key="history_topic")
    num_scenes = st.slider("Number of scenes", 3, 10, 6, key="history_scenes")

    if st.button("Generate video", key="history_run"):
        if not topic:
            st.warning("Enter a topic first.")
        else:
            wd = workdir()
            job_id = job_queue.submit_job("history", f"History: {topic[:30]}",
                                           pipeline_history, topic, num_scenes, wd)
            st.session_state["job_ids"].append(job_id)
            st.session_state["history_job_id"] = job_id
            st.success("Job submitted — check the sidebar for status.")

    show_result_if_ready("history_job_id", "history_final.mp4")
