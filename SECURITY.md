# Security

## Reporting vulnerabilities

**Do not open a public GitHub issue for security vulnerabilities.**

Report privately via GitHub's security advisory system:

> https://github.com/forkwright/heurema/security/advisories/new

Include: description, reproduction steps, potential impact, affected version or commit, and any suggested fix.

## Response SLA

| Severity | Acknowledgment | Fix Target |
|----------|----------------|------------|
| Critical (CVSS ≥ 9.0) | 24 hours | 7 days |
| High (CVSS 7.0–8.9) | 48 hours | 14 days |
| Medium (CVSS 4.0–6.9) | 5 days | 30 days |
| Low (CVSS < 4.0) | 10 days | 90 days |

## Scope

**In scope:**

- Memory-safety or unsoundness in the HNSW (`crates/heurema/src/hnsw/engine.rs`) and BM25 (`crates/heurema/src/fts/bm25.rs`) implementations, including snapshot decoding.
- Persistence-backend trust boundary issues — index files crafted to mislead `PersistenceBackend::load_*`.
- Dependency-chain advisories that become exploitable through heurēma's surface.

**Out of scope:**

- Vulnerabilities only present in unrelated upstream dependencies that heurēma neither propagates nor exacerbates; report those upstream.
- Resource exhaustion from caller-supplied parameters (e.g., `k` too large, vector dimensionality too high). Callers bound their own parameters.

## Disclosure

After a fix ships, we publish a GitHub Security Advisory when warranted, with affected versions, fixed version, impact, remediation, and credit to the reporter.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest minor (`0.x`) | Yes |
| Older `0.x` minors | Best effort; no guarantees pre-1.0 |

## Security Standards

heurēma follows the fleet security standards maintained in `crates/basanos/standards/SECURITY.md` in `forkwright/kanon`. In particular:

- `unsafe_code = "forbid"` workspace-wide. The HNSW and BM25 engines are safe Rust, and the lint keeps them that way.
- No silent truncation: no `as` for numeric conversions in production code paths.
- Backend errors are type-erased only at the `PersistenceSource` boundary, so the error chain stays walkable.
