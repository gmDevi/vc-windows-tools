# Submission checklist (progress prize, deadline 30 Sep 2026, 23:59 Pacific)

Everything below except step 3 (the form) is already done in this repository.

## 1. Publish the repository (one-time authorisation is yours)

```bash
gh auth login --web --git-protocol https
```

Then, from `prize/vesuvius/vc-windows-tools`:

```bash
gh repo create vc-windows-tools --public --source . --remote origin --description "Single-GPU, Windows-native ink search for the Vesuvius Challenge" --push
```

Repository: https://github.com/gmDevi/vc-windows-tools (done). The URL is in `report/SUBMISSION.md` (Links line) and
into the form.

## 2. Discord (done: display name `hamiltonian`)

Joined; the handle is in `report/SUBMISSION.md`. Announcing the tool in the `#progress-prizes` or
tooling channel after submission is encouraged by the organizers (early
release counts).

## 3. Form

Submission form: https://docs.google.com/forms/d/e/1FAIpQLScNBMj25FMnphngRG1Ciryv_2_Mkdq2YPJOD9WqPfZExII2iQ/viewform

Paste the sections of `report/SUBMISSION.md` (title, open problem, what it
does, validation, results, why it helps, links). Attach or link
`report/REPORT.md` and the figures in `report/figures`.

## 4. Before submitting

- `git log --oneline` shows the results and report commits.
- `results/` contains the per-render scores and logs of every sweep run.
- The README quick start runs on a clean machine (the demo needs only the
  villa environment and one checkpoint download).
