"""
Minimal job queue for the Streamlit app.

Runs pipeline functions on a background thread pool instead of blocking the
browser tab, so multiple people (or multiple jobs from one person) can queue
work without waiting on each other one-at-a-time in the UI.

This is intentionally simple: an in-process ThreadPoolExecutor + an in-memory
job registry. It persists for as long as the `streamlit run` process is alive,
and is shared across all sessions/users hitting that same process. It does
NOT survive a server restart and does NOT scale across multiple processes/
machines — if you outgrow this (e.g. deploying with multiple workers), swap
it for something like Redis + RQ/Celery, but this is enough for a small team
testing locally or on one shared box.

Important: pipeline functions run on a background thread, so they must be
plain Python (no st.* calls inside them) — return a result or raise on error,
and let the Streamlit tab code handle all UI updates.
"""

import concurrent.futures
import threading
import uuid
import time

# Free hosting has limited CPU/RAM. Keep video jobs strictly sequential.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_jobs = {}
_lock = threading.Lock()


def submit_job(niche: str, label: str, fn, *args, **kwargs) -> str:
    job_id = str(uuid.uuid4())[:8]
    future = _executor.submit(fn, *args, **kwargs)
    with _lock:
        _jobs[job_id] = {
            "niche": niche,
            "label": label,
            "future": future,
            "submitted_at": time.time(),
        }
    return job_id


def get_status(job_id: str) -> str:
    """Returns one of: 'queued', 'running', 'done', 'error', or None if unknown."""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return None
    future = job["future"]
    if future.done():
        return "error" if future.exception() else "done"
    if future.running():
        return "running"
    return "queued"


def get_result(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job and job["future"].done() and not job["future"].exception():
        return job["future"].result()
    return None


def get_error(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job and job["future"].done():
        return job["future"].exception()
    return None


def get_job_info(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return {}
    return {"niche": job["niche"], "label": job["label"], "submitted_at": job["submitted_at"]}
