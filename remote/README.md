# Remote workloads: how to re-attach after a restart

Every remote job is recorded in `instances.json` (this folder) by the launcher that started it. Nothing about a job
lives only in a chat session: the handle in the registry is enough to find it again from a fresh machine, as long
as the provider key files exist.

| provider | key file | find the job again | fetch results |
| --- | --- | --- | --- |
| Kaggle | `%USERPROFILE%\.kaggle\access_token` | `prize/vesuvius/kaggle/.venv/Scripts/kaggle.exe kernels status <kernel>` | `kaggle kernels output <kernel> -p <dir>` (log + `meshes_*` folder) |
| Lightning | `%USERPROFILE%\.lightning_api_key` | `python lightning/lightning_fit.py status` (studio `vesuvius-fit`, jobs run under `nohup` on the studio disk; logs in `~/vesuvius/jobs/`) | `python lightning/lightning_fit.py fetch <job_id> <dir>` |
| Clore | `%USERPROFILE%\.clore_api_key` | `python clore.py orders` | `scp`/`rsync` over the order's SSH port |

Rules the launchers follow:

- Jobs are started detached on the remote side (Kaggle kernels are detached by nature; Lightning jobs run with
  `nohup` and write `<job>.done` with `DONE` or `FAILED`), so a local crash never kills them.
- The registry entry is written before the launcher returns, with the remote log path and the artifact path.
- `status` commands only read; `fetch` marks the entry `fetched`; `stop` refuses while a registered job is running.
- Kaggle kernels must be pushed with `--accelerator NvidiaTeslaT4`: the fitter's PyTorch build has no P100 kernels.

After a restart: `python registry.py list`, then the provider's `status` command for each `running` entry.
