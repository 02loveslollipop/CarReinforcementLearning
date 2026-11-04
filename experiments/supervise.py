"""Resume an interrupted evolution run, including after agent quota resets."""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def alive(pid):
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or not stat.read_text().split(") ", 1)[1].startswith("Z")
    except (ProcessLookupError, FileNotFoundError):
        return False


def complete(state):
    generations = state.get("generations_data", {})
    return (len(generations) == state["generations"] and all(
        str(g) in generations and len(generations[str(g)]["candidates"]) == state["population"]
        and all(c["status"] == "complete" for c in generations[str(g)]["candidates"])
        for g in range(state["generations"])) )


def quota_delay(state):
    resets = []
    for generation in state.get("generations_data", {}).values():
        for candidate in generation["candidates"]:
            if candidate.get("agent_status") != "failed":
                continue
            log = Path(candidate["directory"]) / "logs" / f"agent_attempt_{candidate['attempt']}.log"
            if not log.is_file():
                continue
            text = log.read_text(errors="replace")
            if "usage limit" not in text.lower():
                continue
            match = re.search(r"try again at\s+(\d{1,2}:\d{2}\s*[AP]M)", text, re.I)
            modified = dt.datetime.fromtimestamp(log.stat().st_mtime).astimezone()
            if match:
                clock = dt.datetime.strptime(match.group(1).upper(), "%I:%M %p").time()
                reset = modified.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)
                if reset < modified:
                    reset += dt.timedelta(days=1)
                resets.append(reset.timestamp() + 15)
            else:
                resets.append(modified.timestamp() + 900)
    return max(0, max(resets, default=0) - time.time()) if resets else None


def workers_alive(run_dir):
    """Do not restart while orphaned agents or training containers are active."""
    commands = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, check=True).stdout
    if any("codex exec" in line and str(run_dir) in line for line in commands.splitlines()):
        return True
    runtime = shutil.which("podman") or shutil.which("docker")
    if runtime is None:
        raise RuntimeError("No container runtime is available to verify surviving workers")
    listing = subprocess.run([runtime, "ps", "-q", "--filter", "ancestor=localhost/car-rl-benchmark:latest"],
                             capture_output=True, text=True, check=True).stdout.split()
    if not listing:
        return False
    inspected = subprocess.run([runtime, "inspect", *listing], capture_output=True, text=True)
    if inspected.returncode:
        return True  # A container may have exited between listing and inspection; retry.
    for container in json.loads(inspected.stdout):
        for mount in container.get("Mounts", []):
            source = mount.get("Source")
            if source and Path(source).is_relative_to(run_dir):
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.wait_pid:
        while alive(args.wait_pid):
            time.sleep(10)
    other_failures = 0
    while True:
        while workers_alive(run_dir):
            print("Waiting for surviving workers to finish before resuming.", flush=True)
            time.sleep(10)
        state = json.loads((run_dir / "state.json").read_text())
        if complete(state):
            print("All candidates completed.", flush=True)
            return
        delay = quota_delay(state)
        if delay is not None:
            print(f"Quota reset wait: {delay:.0f} seconds", flush=True)
            time.sleep(delay)
        elif other_failures >= 3:
            print("Stopped after three unsuccessful resumes without a quota error; inspect candidate logs.", flush=True)
            return
        elif other_failures:
            time.sleep(60)
        print("Resuming incomplete run", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "experiments/orchestrate.py"), "--run-dir", str(run_dir)], cwd=ROOT)
        other_failures = 0 if delay is not None else other_failures + 1


if __name__ == "__main__":
    main()
