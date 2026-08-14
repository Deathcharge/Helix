# Security policy

## Historical status

This repository is a preserved, superseded Helix snapshot. It is not supported for
deployment and does not receive feature or compatibility work. Dependency updates and
integrity checks may be applied when they reduce risk without presenting the archive as
an active product.

Do not submit ordinary vulnerability reports whose only premise is that the historical
applications can be deployed. They must not be deployed. If an issue demonstrates that
a maintained Samsarix repository inherited the same vulnerable behavior, report it
privately through that repository's GitHub Security Advisory interface and identify the
shared code path.

## Handling historical material

- Use synthetic credentials and isolated environments only.
- Do not connect historical entry points to production data or third-party accounts.
- Do not infer current Samsarix guarantees from historical documentation or tests.
- Treat dependency installation and tests as provenance checks, not release evidence.

The repository history is retained intentionally. Potential credential or personal-data
exposure in history should be reported privately to the owner rather than opened as a
public issue.

`Helix.env` is intentionally absent from the current tree. Removing a file from the tip
does not erase it from Git history or invalidate any value it may have contained. Treat
historical values as exposed: rotate them with their issuers and review deployment-secret
stores. Do not paste suspected values into issues, pull requests, logs, or chat.
