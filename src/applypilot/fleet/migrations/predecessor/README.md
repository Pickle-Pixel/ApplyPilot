# Fleet migration predecessor byte bundle

Each file here is named for a Git blob id and contains the **exact committed
bytes** of a predecessor artifact declared in `../manifest-v1.json`
(`predecessor.files[].git_blob`). Together they are the canonical, content-
addressed provenance bundle for the pinned predecessor commit/tree.

They exist so `applypilot.fleet.migrator.verify_predecessor` can prove the exact
predecessor bytes **without a `.git` directory** -- a release archive strips
`.git`, so the object store is unavailable and a bare (commit, tree, blob) tuple
would otherwise be unverifiable. The verifier requires:

- the declared (commit, tree, files) tuple to match the pinned published tuple
  (archive) or the real object store (checkout), and
- every carried artifact to be byte-for-byte exact: `sha1("blob "+len+"\0"+bytes)`
  must equal the artifact's name. Prefix-only, reconstructed, normalized, or
  missing bytes all fail closed.

These are opaque bytes (`-text` in `.gitattributes`) and must never be
normalized: a single CRLF rewrite would change the blob id and break the pin.
