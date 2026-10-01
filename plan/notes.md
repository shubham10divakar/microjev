# Micro-Jev — working notes

Newest first. Decisions, deviations from the design doc, and things to check later.

## 2026-10-01 — start

- Repo: `code_repo/microjev/` with its own `git init`; added `microjev/` to
  `code_repo/.git/info/exclude` so the outer nano-jev repo ignores it.
- **Deviation:** design §9 names the folder `code_repo/jev_core/`. Code lives in `microjev/`
  instead (where the design doc is). The shared package is still called `jevcore`, so Tiny-Jev
  can import it later or the package can be moved without renames.
- GPU busy: implement only. Tests use a tiny random ModernBERT config on CPU.
