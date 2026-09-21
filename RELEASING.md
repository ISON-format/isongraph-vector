# Releasing

One tag publishes six packages to four registries and cuts a GitHub release.
This page is the whole procedure, the one-time setup each registry needs, and
what to do when a publish fails halfway.

The mechanics live in
[`.github/workflows/release.yml`](https://github.com/ISON-format/isongraph-vector/blob/main/.github/workflows/release.yml).
Nothing is uploaded until the tag has been checked against every manifest,
the full CI suite has passed on exactly that ref, and every artifact has been
built. The four publishes then run in parallel, each behind its own
environment so any of them can require approval, and each tolerates a rerun.

## Cutting a release

1. **Bump the version everywhere.** Nine files carry it, in six syntaxes:

   ```text
   pyproject.toml                                            version = "…"
   isongraph_vector-py/__init__.py                           __version__ = "…"
   isongraph-vector-rs/Cargo.toml                            version = "…"
   isongraph-vector-ts/package.json                          "version": "…"
   isongraph-vector-js/package.json                          "version": "…"
   isongraph-vector-csharp/src/IsonGraph.Vector/…csproj       <Version>…</Version>
   isongraph-vector-csharp/src/IsonGraph.Vector.Sqlite/…csproj
   isongraph-vector-cpp/CMakeLists.txt                       project(… VERSION …)
   isongraph-vector-cpp/include/isongraph_vector.hpp         VERSION = "…"
   ```

   Then prove they agree:

   ```bash
   python scripts/release/check_versions.py 1.0.1
   ```

   The release workflow runs the same check first and refuses a tag that any
   manifest disagrees with. A lagging file is the default outcome of bumping
   by hand, and the registries would accept the mismatch.

2. **Write the CHANGELOG section.** `## [1.0.1] - YYYY-MM-DD` at the top of
   [`CHANGELOG.md`](https://github.com/ISON-format/isongraph-vector/blob/main/CHANGELOG.md),
   with a `[1.0.1]:` link at the bottom. The release notes are lifted from
   this section verbatim; the version check fails if it is missing.

3. **Merge to `main` and let CI go green.** The release workflow reruns CI
   on the tag regardless, but a red `main` is not something to tag.

4. **Tag and push.**

   ```bash
   git tag -a v1.0.1 -m "isongraph-vector 1.0.1"
   git push origin v1.0.1
   ```

   The tag must be on `main` - the workflow checks - and must match
   `v<major>.<minor>.<patch>` exactly. Nothing else triggers it.

5. **Approve the environments**, if you have given them required reviewers
   (below). Each publish waits for its own approval, so PyPI can go out while
   you are still deciding about NuGet.

6. **Check the registries.** The workflow's job summary links each one. The
   badges in the READMEs resolve as soon as the registry indexes the version,
   usually within minutes.

## What the workflow does, in order

| Job | Gate | Does |
| --- | --- | --- |
| `verify` | — | Ref is a tag on `main`; nine manifests declare the tagged version; CHANGELOG has the section. |
| `test` | `verify` | The whole of [`ci.yml`](https://github.com/ISON-format/isongraph-vector/blob/main/.github/workflows/ci.yml), on the tag - all six suites, packaging, the cross-port interchange check. |
| `build` | `verify` | sdist and wheel, both `.nupkg`, both npm tarballs, the `.crate`, a C++ header tarball, and `SHA256SUMS` over all of them. Uploaded as one workflow artifact so every publish uses the same bytes. |
| `publish-pypi` | all three | `pypa/gh-action-pypi-publish`, trusted publishing, `skip-existing`. |
| `publish-nuget` | all three | `NuGet/login` exchanges the OIDC token for a short-lived key; pushes core then SQLite, `--skip-duplicate`. |
| `publish-npm` | all three | `npm publish --provenance` for both tarballs, skipping any version already on the registry. |
| `publish-crates` | all three | `rust-lang/crates-io-auth-action`, then `cargo publish`, skipping if `cargo info` already sees the version. |
| `github-release` | `test`, `build`, the publishes | Creates `v<version>` with the CHANGELOG section as notes, every artifact and `SHA256SUMS` attached, and a line recording each publish's outcome. Runs even if a publish failed, so the release exists for whatever did ship - but not if CI failed, because then nothing was published and the notes would advertise installs that do not resolve. |

## One-time setup: trusted publishing

Every registry here supports OpenID Connect trusted publishing: the registry
is told to trust this repository's `release.yml`, and the workflow authenticates
with a short-lived token GitHub mints per run. No long-lived secret is stored
in the repository. Configure each once; the values are the same everywhere:

```text
Repository owner   ISON-format
Repository name    isongraph-vector
Workflow filename  release.yml
Environment        (per registry, below)
```

The environments are created automatically the first time the workflow
references them. To require a human approval before any publish, open
**Settings → Environments** on GitHub, pick the environment, and add yourself
under *Required reviewers*. That is the whole gate.

### PyPI — environment `pypi`

Works for a first publish. On <https://pypi.org/manage/account/publishing/>,
add a **pending publisher** with the project name `isongraph-vector` and the
values above. PyPI creates the project on the first trusted upload. No
fallback is provided; this is the documented first-publish path.

### NuGet — environment `nuget`

Works for a first publish. Trust is configured on the account rather than
the package: on <https://www.nuget.org/account/trustedpublishing>, add a
policy with the values above. Then set a repository **variable** (not
secret) `NUGET_USER` to the nuget.org username the policy was created under;
`NuGet/login` needs it to know which account's policy to match.

Fallback: a `NUGET_API_KEY` secret. The workflow uses it when the login step
produced nothing.

### npm — environment `npm`, two packages

Trusted publishers are configured per package, on the package's settings
page - which does not exist until the package does. For the first release,
set an `NPM_TOKEN` secret (a granular access token with publish rights and
2FA bypass for automation). Once `isongraph-vector-ts` and
`isongraph-vector-js` exist, open each package's *Settings → Trusted
publisher* on npmjs.com, enter the values above, and delete the secret. From
then on the workflow publishes with OIDC and attaches a provenance
attestation either way.

### crates.io — environment `crates-io`

The same shape as npm: trusted publishing is configured on the crate's
settings page, which exists only after the first publish. For the first
release, set a `CARGO_REGISTRY_TOKEN` secret (an API token scoped to
`publish-new` and `publish-update`). Once `isongraph-vector-rs` exists, open
its *Settings → Trusted Publishing* on crates.io, enter the values above, and
delete the secret.

### C++ — no registry

The release attaches `isongraph-vector-cpp-<version>.tar.gz`: the port
directory minus its build tree, so `add_subdirectory` works on the unpacked
result. CMake users can pin the tag directly instead:

```cmake
include(FetchContent)
FetchContent_Declare(isongraph_vector
    GIT_REPOSITORY https://github.com/ISON-format/isongraph-vector.git
    GIT_TAG        v1.0.0
    SOURCE_SUBDIR  isongraph-vector-cpp)
FetchContent_MakeAvailable(isongraph_vector)
target_link_libraries(your_target PRIVATE isongraph_vector)
```

## When a publish fails partway

Every publish skips a version the registry already has, so the recovery for
any partial failure is the same: fix the cause, then rerun from the tag.

1. Open **Actions → Release → Run workflow**.
2. Under *Use workflow from*, select the **tag** - not the branch it points
   at. The workflow refuses a branch ref.
3. Tick **Publish**. Leave it unticked to rebuild everything and upload
   nothing, which is the dry run.

A registry that already has the version is skipped; the ones that failed are
retried; the GitHub release is updated in place rather than duplicated.

A release that must not go out at all is withdrawn on each registry
separately - `yank` on crates.io, `unpublish` within 72 hours on npm,
*unlist* on NuGet, *yank* on PyPI - and the tag deleted. None of that is
automated, deliberately.

## Pre-publish rehearsal

To exercise everything short of the uploads, dispatch the workflow from a tag
with **Publish** unticked. `verify`, `test` and `build` run in full and the
artifacts are attached to the workflow run for inspection. The four publish
jobs and the GitHub release are skipped.
