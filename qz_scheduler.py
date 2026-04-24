#!/usr/bin/env python3
"""
QZ Platform Job Scheduler for Libero Training

Parallel-submits training scripts to QZ platform with priority 3 (low),
monitors job status every 5 minutes, and auto-resubmits killed jobs.

Usage:
    python scripts/qz_scheduler.py

    # Background:
    nohup python scripts/qz_scheduler.py > qz_scheduler.log 2>&1 &
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "username": "", # TODO
    "password": "", # TODO
    "logic_compute_group_id": "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7",
    "project_id": "project-97ab58cb-3162-4d0e-9137-1299d6cdea25",
    "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
    "spec_id": "45ab2351-fc8a-4d50-a30b-b39a5306c906",
    "image": "", # TODO
    "image_type": "SOURCE_PRIVATE",
    "framework": "pytorch",
    "task_priority": 3,
    "shm_gi": 0,
    "poll_interval": 300,
}

BASE_URL = "https://qz.sii.edu.cn"
WORK_DIR = "/inspire/hdd/project/robot3d/mazipei-253107140027/UniVLA"

SCRIPTS = [
    "train_libero_video_dit_chunk-20.sh",
    "train_libero_video_dit_chunk-5.sh",
    "train_libero_video_dit_num_act-32.sh",
    "train_libero_video_dit_num_act-8.sh",
    "train_libero_video_dit_size-B.sh",
    "train_libero_video_dit_size-S.sh",
    "train_libero_video_dit_vision_5e-2.sh",
    "train_libero_video_dit_vision_5e0.sh",
]

RUNNING_STATES = {"job_running", "job_pending", "job_starting", "job_queued", "job_queuing", "job_created", "job_creating"}
COMPLETED_STATES = {"job_completed", "job_succeeded", "job_success"}


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def curl_post(url, payload, token=None, timeout=30):
    """Use subprocess curl for reliable timeout behavior."""
    headers = ["--header", "Content-Type: application/json"]
    if token:
        headers += ["--header", f"Authorization: Bearer {token}"]
    cmd = [
        "curl", "--noproxy", "*", "-sk",
        "--connect-timeout", "10",
        "--max-time", str(timeout),
        "--location", "--request", "POST", url,
        *headers,
        "--data-raw", json.dumps(payload),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr[:200]}")
    return json.loads(result.stdout)


class QZClient:
    def __init__(self, config):
        self.config = config
        self.token = None
        self.token_time = 0

    def _ensure_token(self):
        if self.token and (time.time() - self.token_time) < 3000:
            return
        log("Requesting new access token...")
        data = curl_post(
            f"{BASE_URL}/auth/token",
            {"username": self.config["username"], "password": self.config["password"]},
        )
        if data.get("code") != 0:
            raise RuntimeError(f"Token request failed: {data}")
        self.token = data["data"]["access_token"]
        self.token_time = time.time()
        log("Token acquired successfully.")

    def create_job(self, name, command):
        self._ensure_token()
        payload = {
            "name": name,
            "logic_compute_group_id": self.config["logic_compute_group_id"],
            "project_id": self.config["project_id"],
            "auto_fault_tolerance": False,
            "framework": self.config["framework"],
            "command": command,
            "task_priority": self.config["task_priority"],
            "workspace_id": self.config["workspace_id"],
            "framework_config": [{
                "image": self.config["image"],
                "image_type": self.config["image_type"],
                "instance_count": 1,
                "shm_gi": self.config["shm_gi"],
                "spec_id": self.config["spec_id"],
            }],
        }
        data = curl_post(f"{BASE_URL}/openapi/v1/train_job/create", payload, self.token)
        if data.get("code") != 0:
            raise RuntimeError(f"Create job failed: {data}")
        return data["data"]["job_id"]

    def get_job_detail(self, job_id):
        self._ensure_token()
        data = curl_post(f"{BASE_URL}/openapi/v1/train_job/detail", {"job_id": job_id}, self.token)
        if data.get("code") != 0:
            raise RuntimeError(f"Get job detail failed: {data}")
        return data["data"]

    def stop_job(self, job_id):
        self._ensure_token()
        return curl_post(f"{BASE_URL}/openapi/v1/train_job/stop", {"job_id": job_id}, self.token)


class JobTracker:
    def __init__(self, script_name):
        self.script_name = script_name
        self.job_id = None
        self.status = "NOT_SUBMITTED"
        self.retry_count = 0
        self.completed = False


def print_summary(trackers):
    log("=" * 60)
    log("Job Summary:")
    log(f"  {'Script':<45} {'Status':<15} {'Retries'}")
    log("-" * 60)
    for t in trackers:
        status = "COMPLETED" if t.completed else t.status
        log(f"  {t.script_name:<45} {status:<15} {t.retry_count}")
    log("=" * 60)


def submit_job(client, tracker):
    script_path = f"scripts/simulator/libero/{tracker.script_name}"
    command = f"cd {WORK_DIR} && bash {script_path}"
    job_name = tracker.script_name.replace(".sh", "").replace("_", "-")
    job_id = client.create_job(job_name, command)
    tracker.job_id = job_id
    return job_id


def main():
    client = QZClient(CONFIG)
    trackers = [JobTracker(s) for s in SCRIPTS]

    shutdown = False

    def handle_signal(signum, frame):
        nonlocal shutdown
        log(f"Received signal {signum}, shutting down gracefully...")
        shutdown = True
        print_summary(trackers)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Submit all jobs
    log(f"Submitting {len(SCRIPTS)} training jobs (priority={CONFIG['task_priority']})...")
    for tracker in trackers:
        try:
            job_id = submit_job(client, tracker)
            tracker.status = "SUBMITTED"
            log(f"  Submitted: {tracker.script_name} -> job_id={job_id}")
        except Exception as e:
            log(f"  FAILED to submit {tracker.script_name}: {e}")
            tracker.status = "SUBMIT_FAILED"
        time.sleep(1)

    # Monitor loop
    log(f"Entering monitor loop (interval={CONFIG['poll_interval']}s)...")
    while not shutdown:
        time.sleep(CONFIG["poll_interval"])
        if shutdown:
            break

        log("Checking job statuses...")
        all_done = True

        for tracker in trackers:
            if tracker.completed:
                continue

            if tracker.job_id is None:
                try:
                    job_id = submit_job(client, tracker)
                    tracker.status = "SUBMITTED"
                    tracker.retry_count += 1
                    log(f"  Re-submitted (retry #{tracker.retry_count}): {tracker.script_name} -> job_id={job_id}")
                except Exception as e:
                    log(f"  FAILED to re-submit {tracker.script_name}: {e}")
                all_done = False
                time.sleep(1)
                continue

            try:
                detail = client.get_job_detail(tracker.job_id)
                status = detail.get("status", "unknown")
                tracker.status = status

                if status in COMPLETED_STATES:
                    tracker.completed = True
                    log(f"  COMPLETED: {tracker.script_name} (job_id={tracker.job_id})")
                elif status in RUNNING_STATES:
                    all_done = False
                else:
                    log(f"  Job {tracker.script_name} status={status}, resubmitting...")
                    try:
                        job_id = submit_job(client, tracker)
                        tracker.status = "RESUBMITTED"
                        tracker.retry_count += 1
                        log(f"  Re-submitted (retry #{tracker.retry_count}): {tracker.script_name} -> job_id={job_id}")
                    except Exception as e:
                        log(f"  FAILED to re-submit {tracker.script_name}: {e}")
                    all_done = False
            except Exception as e:
                log(f"  Error checking {tracker.script_name}: {e}")
                all_done = False

            time.sleep(1)

        if all_done:
            log("All jobs completed!")
            break

    print_summary(trackers)


if __name__ == "__main__":
    main()
