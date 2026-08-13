# Helix — historical source archive

This repository preserves an early Helix aggregate for provenance and research. It is
superseded by the maintained Samsarix repositories and is **not a production-ready,
supported, or deployable product**.

## Safety boundary

- Do not deploy the included FastAPI, Discord, Railway, Streamlit, or agent entry
  points as maintained Samsarix services.
- Do not use the historical environment, deployment, or architecture documents as
  current operational guidance.
- `requirements.txt` exists only to make security review and historical replay
  reproducible. A successful install or integrity check does not create a support
  commitment.
- Never place live credentials in this repository. Treat every historical integration
  surface as untrusted until it is reviewed against a maintained Samsarix component.

## Maintained successors

Current work lives in focused repositories under the
[Deathcharge organization](https://github.com/Deathcharge), including
[`samsarix-core`](https://github.com/Deathcharge/samsarix-core) and
[`samsarix-agent-framework`](https://github.com/Deathcharge/samsarix-agent-framework).
The former Helix branding is retained here because changing historical identity would
make provenance harder to follow.

## What is preserved

The tree contains competing application entry points, agent experiments, protocol
prototypes, deployment notes, and tests from the earlier Helix era. Git history and
the rollback tag `rollback/pre-historical-security-20260813` preserve the pre-hardening
snapshot, including the removed case-colliding duplicate `readme.md`.

## Integrity checks

The repository workflow performs a Python 3.11 dependency install, syntax compilation,
the surviving test suite, and a dependency audit. These checks are an archival safety
measure only; they do not validate production behavior, third-party integrations,
deployment configuration, or compatibility with maintained Samsarix products.

See [SECURITY.md](SECURITY.md) before inspecting or replaying the code.
