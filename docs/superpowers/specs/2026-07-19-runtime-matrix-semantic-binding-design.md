# Runtime Matrix Semantic Binding Design

## Goal

Make the Runtime command-authority matrix fail closed when a tracked launcher changes behavior, when a scheduled task changes its executable, arguments, or principal, or when PowerShell introduces a generated systemd unit the scanner cannot safely parse.

## Design

Every discovered surface binds the SHA-256 of its committed source blob in `canonicalTarget`. Scheduled-task surfaces additionally bind any statically extractable `-Execute`, `-Argument`, registration `-User`, and registration `-Principal` values. Existing dynamic expressions remain supported only through an explicit deterministic dynamic marker while the source-blob digest binds their exact implementation. Duplicate or ambiguous fields fail closed. Commented scheduled-task examples are excluded before matching so they cannot create phantom rows.

PowerShell systemd generation is not interpreted. Any committed PowerShell that writes a `.service` or `.timer` destination fails discovery as an unsupported generator, preventing silent omission. Shell-generated units continue using the existing structural parser; common alternative shell systemd writers and inline launchd writers fail closed until a structural parser supports them. Generated-unit activation binds an exact unit token rather than accepting substring matches.

Repository and matrix paths are resolved before containment and receipt-relative-path checks, so a repository-relative `--matrix` path works while an out-of-repository matrix remains rejected.

## Tests

Focused tests prove launcher source edits and task executable/argument/principal edits create stale matrix rows, existing dynamic task forms receive deterministic markers, ambiguous task bindings fail, comments do not create phantom rows, unsupported service generators fail closed, activation tokens match exactly, and relative matrix CLI invocation succeeds. The checked-in matrix is regenerated from the exact repaired tree, followed by focused pytest and the exact verifier.
