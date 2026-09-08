r"""Registry of remote workloads, so any later session (or a fresh machine) can re-attach to running jobs.
State lives in prize/remote/instances.json, one entry per job:
  {"id": "<provider>:<name>", "provider": "kaggle|lightning|clore", "handle": {...how to find it again...},
   "job": {...what it runs...}, "status": "running|done|failed|stopped", "started": iso, "updated": iso, "notes": ""}
  registry.py list                      print all entries
  registry.py set <id> <json-fields>    add or update an entry (fields merged)
  registry.py status <id> <status>      quick status update
"""
import json, os, sys
from datetime import datetime, timezone
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances.json")


def load():
    return json.load(open(PATH, encoding="utf-8")) if os.path.exists(PATH) else {}


def save(reg):
    tmp = PATH + ".tmp"
    json.dump(reg, open(tmp, "w", encoding="utf-8"), indent=1, sort_keys=True)
    os.replace(tmp, PATH)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_entry(id_, **fields):
    reg = load(); e = reg.get(id_, {"id": id_, "started": now()})
    e.update(fields); e["updated"] = now(); reg[id_] = e; save(reg); return e


def set_status(id_, status, note=None):
    reg = load(); e = reg.get(id_)
    if not e: raise SystemExit(f"unknown id {id_}")
    e["status"] = status; e["updated"] = now()
    if note: e["notes"] = (e.get("notes", "") + " | " + note).strip(" |")
    save(reg); return e


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "list":
        for e in sorted(load().values(), key=lambda e: e["updated"], reverse=True):
            print(f"{e['status']:8} {e['id']:45} {e['updated']}  {json.dumps(e.get('job', {}))[:90]}  {e.get('notes', '')[:80]}")
    elif a[0] == "set":
        print(set_entry(a[1], **json.loads(a[2])))
    elif a[0] == "status":
        print(set_status(a[1], a[2], a[3] if len(a) > 3 else None))
    else:
        print(__doc__)
