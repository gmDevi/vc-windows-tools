r"""Run spiral fits on Lightning AI studios, re-attachable from any session via prize/remote/instances.json.
Key: %USERPROFILE%\.lightning_api_key (never printed). Teamspace/org fixed below.

  lightning_fit.py start <SCROLL> <VOLUME> <LASAGNA_RUN> <Z0> <Z1> <CW|ACW> <NWIND> <STEPS> <UMB_URL> [tag] [machine]
        starts (or reuses) the studio, uploads fit_job.sh, launches it detached (nohup), records the job in the registry
  lightning_fit.py status [job_id]        studio status + tail of the job log for one or all lightning jobs in the registry
  lightning_fit.py fetch <job_id> <dst>   download meshes_<job>.tgz when the job is DONE, mark the registry entry
  lightning_fit.py stop                   stop the studio (only when no job is running)
"""
import os, sys, json, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import registry
os.environ["LIGHTNING_API_KEY"] = open(os.path.expanduser("~/.lightning_api_key"), encoding="utf-8").read().strip()
from lightning_sdk import Studio, Machine, Teamspace
ORG, TEAMSPACE, STUDIO = "mdevincenzis-org", "api-credential-management-project", "vesuvius-fit"
REMOTE_ROOT = "vesuvius"  # relative to the studio root /teamspace/studios/this_studio
ABS_ROOT = "/teamspace/studios/this_studio/vesuvius"


def studio(create=False):
    return Studio(name=STUDIO, teamspace=TEAMSPACE, org=ORG, create_ok=create)


def ensure_running(s, machine):
    st = str(s.status)
    if "Running" not in st:
        print("studio status:", st, "-> starting on", machine, flush=True)
        s.start(getattr(Machine, machine))
        for _ in range(60):
            time.sleep(10)
            if "Running" in str(s.status): break
    print("studio:", s.name, "|", s.status, "|", s.machine, flush=True)


def run(s, cmd):
    return s.run(cmd)


def cmd_start(a):
    scroll, volume, lrun, z0, z1, sense, nwind, steps, umb = a[:9]
    tag = a[9] if len(a) > 9 else f"{sense.lower()}{int(steps)//1000}k_w{nwind}"
    machine = a[10] if len(a) > 10 else "T4"
    job = f"{scroll}_{tag}"; jid = f"lightning:{STUDIO}:{job}"
    s = studio(create=True); ensure_running(s, machine)
    s.upload_file(os.path.join(HERE, "fit_job.sh"), f"{REMOTE_ROOT}/fit_job.sh")
    env = f"SCROLL={scroll} VOLUME={volume} LASAGNA_RUN={lrun} Z0={z0} Z1={z1} SENSE={sense} NWIND={nwind} STEPS={steps} UMB_URL='{umb}' TAG={tag}"
    out = run(s, f"mkdir -p {ABS_ROOT}/jobs && cd {ABS_ROOT} && ls -la fit_job.sh && chmod +x fit_job.sh && ({env} nohup bash ./fit_job.sh > jobs/{job}.nohup 2>&1 &) && sleep 3 && pgrep -f 'fit_job.sh' | head -3")
    print("launched, pids:", out.strip()[:80])
    registry.set_entry(jid, provider="lightning", status="running",
                       handle={"org": ORG, "teamspace": TEAMSPACE, "studio": STUDIO, "machine": machine, "log": f"{ABS_ROOT}/jobs/{job}.log", "done": f"{ABS_ROOT}/jobs/{job}.done", "artifact": f"{ABS_ROOT}/jobs/meshes_{job}.tgz"},
                       job={"scroll": scroll, "volume": volume, "z0": int(z0), "z1": int(z1), "sense": sense, "nwind": int(nwind), "steps": int(steps), "tag": tag, "umbilicus": umb})
    print("registered", jid)


def cmd_status(a):
    s = studio(); print("studio:", s.name, "|", s.status, "|", s.machine)
    if "Running" not in str(s.status):
        print("studio not running; job logs are on its disk and reappear when it starts"); return
    reg = registry.load()
    ids = [a[0]] if a else [k for k in reg if k.startswith("lightning:")]
    for jid in ids:
        e = reg[jid]; log, done = e["handle"]["log"], e["handle"]["done"]
        out = run(s, f"echo DONEFILE:$(cat {done} 2>/dev/null); tail -n 6 {log} 2>/dev/null | cut -c1-160; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null")
        print(f"== {jid} (registry: {e['status']})\n{out}")
        if "DONEFILE:DONE" in out and e["status"] == "running": registry.set_status(jid, "done", "job wrote DONE")
        if "DONEFILE:FAILED" in out and e["status"] == "running": registry.set_status(jid, "failed", "job wrote FAILED")


def cmd_fetch(a):
    jid, dst = a[0], a[1]; e = registry.load()[jid]; s = studio(); ensure_running(s, e["handle"]["machine"])
    art = e["handle"]["artifact"].replace("/teamspace/studios/this_studio/", "")
    os.makedirs(dst, exist_ok=True); local = os.path.join(dst, os.path.basename(art))
    s.download_file(art, local); print("downloaded", local, os.path.getsize(local), "bytes")
    subprocess.run(["tar", "xzf", local, "-C", dst], check=True); print("extracted into", dst)
    registry.set_status(jid, "fetched", f"meshes in {dst}")


def cmd_stop(a):
    s = studio(); running = [k for k, e in registry.load().items() if k.startswith("lightning:") and e["status"] == "running"]
    if running and not a: print("refusing to stop: running jobs", running, "(pass --force)"); return
    s.stop(); print("stopped", s.name)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"start": cmd_start, "status": cmd_status, "fetch": cmd_fetch, "stop": cmd_stop}.get(a[0] if a else "", lambda _: print(__doc__))(a[1:])
