# Changelog

## [0.3.0](https://github.com/forkwright/heurema/compare/v0.2.0...v0.3.0) (2026-09-26)


### Features

* **heurema:** add phase 02 index record and operation identity types ([#55](https://github.com/forkwright/heurema/issues/55)) ([bf833e4](https://github.com/forkwright/heurema/commit/bf833e41168e13580b33d4ab84879b753c54f70f)), closes [#30](https://github.com/forkwright/heurema/issues/30)
* **heurema:** implement navigable HNSW graph ([#50](https://github.com/forkwright/heurema/issues/50)) ([21e5470](https://github.com/forkwright/heurema/commit/21e5470f9bbf67874805c2bebe9364c964916bde))
* **heurema:** implement simple BM25 retrieval ([#48](https://github.com/forkwright/heurema/issues/48)) ([538969b](https://github.com/forkwright/heurema/commit/538969b517703878dc2a5bba2cb606c1276f9efd))
* **heurema:** validate phase 02 lifecycle operations before any write ([#57](https://github.com/forkwright/heurema/issues/57)) ([da6c324](https://github.com/forkwright/heurema/commit/da6c324069f6b11a06cede51e90f4931ea52eeee)), closes [#30](https://github.com/forkwright/heurema/issues/30)
* **persistence:** version individual index snapshots ([#51](https://github.com/forkwright/heurema/issues/51)) ([af7eeae](https://github.com/forkwright/heurema/commit/af7eeae7f2e750ad80408ea0d8487225bc66411f))
* **workspace:** stage and atomically publish phase 02 index versions ([#59](https://github.com/forkwright/heurema/issues/59)) ([85032ca](https://github.com/forkwright/heurema/commit/85032ca243c999422cc948065ba81950e044ee4e)), closes [#30](https://github.com/forkwright/heurema/issues/30)

## [0.2.0](https://github.com/forkwright/heurema/compare/v0.1.3...v0.2.0) (2026-08-16)


### Features

* **workspace:** implement PersistenceBackend with thesauros and atmis ([#24](https://github.com/forkwright/heurema/issues/24)) ([62e26ed](https://github.com/forkwright/heurema/commit/62e26eddcca51d3d2cd6becad85401d151ce0f9f))

## [0.1.3](https://github.com/forkwright/heurema/compare/v0.1.2...v0.1.3) (2026-08-03)


### Bug Fixes

* **heurema:** compose index result contracts with RRF ([#14](https://github.com/forkwright/heurema/issues/14)) ([0fd437c](https://github.com/forkwright/heurema/commit/0fd437c1f51c3f1a9d346d7d6975671d2f22d1c3)), closes [#2](https://github.com/forkwright/heurema/issues/2)
* **release:** match path-deps by filter so new crates stay covered ([#12](https://github.com/forkwright/heurema/issues/12)) ([02885c1](https://github.com/forkwright/heurema/commit/02885c15a7c6ec1252f90e92f9938a2cec8072e2))

## [0.1.2](https://github.com/forkwright/heurema/compare/v0.1.1...v0.1.2) (2026-07-29)


### Bug Fixes

* **release:** keep Cargo.lock in lockstep with the workspace version ([#10](https://github.com/forkwright/heurema/issues/10)) ([22fc2cf](https://github.com/forkwright/heurema/commit/22fc2cfb43107d88769fd094e39b2cf726b5cdf8)), closes [#7](https://github.com/forkwright/heurema/issues/7)

## [0.1.1](https://github.com/forkwright/heurema/compare/v0.1.0...v0.1.1) (2026-07-08)


### Features

* **heurema:** Phase 1 scaffold + crate extraction from kanon ([#1](https://github.com/forkwright/heurema/issues/1)) ([d5d9085](https://github.com/forkwright/heurema/commit/d5d9085328cf2234646cdbd4dd036575edbd82a1))


### Bug Fixes

* resolve open audit findings (RRF determinism + typed errors) + Tier-U CI ([#4](https://github.com/forkwright/heurema/issues/4)) ([28b0942](https://github.com/forkwright/heurema/commit/28b09421bb290ffe0163c998acaafcd87f18ef10))

## Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

release-please appends release entries below this header on first run.
