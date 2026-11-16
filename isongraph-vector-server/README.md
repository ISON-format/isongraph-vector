# isongraph-vector-server

A multi-tenant HTTP service, a tenant-sharding router and snapshot read replicas around
[`isongraph-vector`](../README.md). The library is deliberately an embedded component -
its roadmap lists "multi-tenant access control and a service layer" as out of scope - so this
is a **separate package**: the library is used unmodified, as a dependency.

**Status: 0.1.0, alpha.** It has real tests (see [Testing](#testing)), but it is new code and has not
been run in production. Read [What is NOT provided](#what-is-not-provided) before relying on it.

The only runtime dependency is the library itself (and through it `ison-graph`). The HTTP server,
router, replication and CLI are standard library only. `numpy` is optional (vectorised search and the
opt-in IVF index come from the library).

## What it is

```
                      clients (HTTPS, via your TLS proxy)
                                   |
                         +---------v---------+
                         |  igv-server router |   stateless; tenant -> shard by rendezvous hash
                         +----+---------+----+   (reads may go to a replica; writes never retried)
                              |         |
                 +------------v--+   +--v------------+
                 | shard s0      |   | shard s1      |      one primary per shard (no failover)
                 | primary       |   | primary       |
                 | tenants A,C.. |   | tenants B,D.. |      each tenant lives on exactly one shard
                 +------+--------+   +---------------+
                        | pull snapshots (generation + sha256 + graph digest)
                 +------v--------+
                 | read replica  |      serves a *saved generation*, read-only, bounded staleness
                 +---------------+
```

* **Tenancy.** Every request is bound to a tenant by its API key. The tenant is derived from the
  verified key and from nothing else (no header, query parameter, body field or path segment can
  select or override it). Each tenant has its own graphs, quotas, rate limit and directory.
* **API.** JSON over HTTP: create/delete graphs; upsert/patch/delete nodes and edges; semantic search,
  `multi_hop`, `path`, `subgraph`, `similar`; declarative filters and edge weights; stale-embedding
  detection and re-embedding; usage, health and metrics. See [API](#api).
* **Durability.** Atomic, checksummed, digest-verified snapshot *generations* with automatic fallback;
  crash-tested with real `SIGKILL`.
* **Scale-out.** Hash-shard tenants across nodes behind a stateless router; add snapshot-based read
  replicas. See [Scale-out and consistency](#scale-out-and-consistency) for exactly what that does and
  does not give you.

## Quick start

```bash
pip install ./isongraph-vector-server          # from a checkout; brings the library too
export IGV_DATA_DIR=./igv-data

igv-server tenant create acme --quota max_nodes=50000
igv-server key create acme --role rw --label ci      # prints the key ONCE, stores only a salted hash
#   igv_acme.3f9a....

cat > config.json <<'EOF'
{ "data_dir": "./igv-data", "port": 8080,
  "encoder": {"kind": "mock", "dimension": 64} }      # "mock" is plumbing only; see config.example.json
EOF
igv-server serve --config config.json &

KEY=igv_acme.3f9a...
curl -s -X POST localhost:8080/v1/graphs -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"name":"kb"}'
curl -s -X POST localhost:8080/v1/graphs/kb/nodes -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"nodes":[{"type":"person","id":1,"properties":{"name":"Alice","description":"builds payment APIs"}}]}'
curl -s -X POST localhost:8080/v1/graphs/kb/search -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"query":"who builds APIs?","top_k":3}'
```

Docker: `docker build -f isongraph-vector-server/Dockerfile -t igv-server .` from the repository root
(see the comments in the Dockerfile; the image has no torch unless you pass
`--build-arg WITH_SENTENCE_TRANSFORMERS=1`).

## API

All paths are under `/v1` and need `Authorization: Bearer <key>` unless noted. Bodies are JSON
(`Content-Type: application/json`), validated against an explicit schema (`schemas.py`): unknown fields,
duplicate keys, `NaN`/`Infinity`, nested property values and invalid UTF-8 are rejected.

| Method and path | Purpose |
| --- | --- |
| `GET /healthz`, `GET /readyz` (no key) | liveness; readiness (`503` until startup verification finished, and on a replica past its staleness bound) |
| `GET /metrics` (operator token) | Prometheus text, aggregate only. Exists only if `IGV_METRICS_TOKEN` is set |
| `GET /v1/usage` | this tenant's quotas and usage |
| `GET /v1/graphs`, `POST /v1/graphs` | list (paginated) / create `{name, embedding:{mode,dimension}, embed_fields, ann}` |
| `GET /v1/graphs/{g}`, `DELETE /v1/graphs/{g}` | info (counts, generation, `snapshot_digest`) / delete |
| `POST /v1/graphs/{g}/save` | force a snapshot now |
| `GET /v1/graphs/{g}/nodes` | list nodes (`type`, `limit`, `cursor`) |
| `POST /v1/graphs/{g}/nodes` | batch upsert (create or **replace**); `nodes/delete` for batch delete |
| `GET/PUT/PATCH/DELETE /v1/graphs/{g}/nodes/{type}/{id}` | one node; `PATCH` merges (`remove:[...]` deletes properties) |
| `GET /v1/graphs/{g}/edges`, `POST .../edges`, `POST .../edges/delete` | list / batch upsert / batch delete |
| `POST /v1/graphs/{g}/search` | semantic search: `query` (text) or `vector`, `top_k`, `threshold`, `node_type`, `filter`, `exact` |
| `POST /v1/graphs/{g}/multi_hop` | seed by similarity, traverse with decay: `rel_type`, `max_hops`, `direction`, `decay`, `blend`, `weight_field`, `filter`, `strict_filter` |
| `POST /v1/graphs/{g}/path`, `.../subgraph`, `.../similar` | `semantic_path`, `semantic_subgraph` (returned as nodes+edges JSON), `similar_to_node` |
| `GET /v1/graphs/{g}/stale`, `POST .../reembed` | model lifecycle (server-mode graphs) |

* **Embedding modes.** `server` (default): the server embeds text with its configured encoder (node
  `properties` named in `embed_fields`, or an explicit `embed_text`). `client`: you create the graph with a
  `dimension` and send `vector`s; text queries are refused. A graph never mixes the two, so a stored vector
  always has a known origin.
* **Filters** are data, never code: `{"types":[...], "where":[{"field":"age","op":"gte","value":30}]}`
  with `eq ne lt lte gt gte in nin exists prefix contains`, ANDed, at most 16 conditions.
* **Ids.** Integer, or a string matching `[A-Za-z0-9_][A-Za-z0-9_.@+=~-]{0,127}`. A string that spells a
  canonical integer *is* that integer (`"5"` and `5` are one node - the library cannot hold both).
* **Pagination.** `limit` (default 100, max `max_page`) and an opaque `cursor`; responses carry
  `next_cursor`. Node cursors are keyset positions; edge and stale cursors are offsets (best effort if the
  graph changes between pages).
* **Errors** always look like `{"error": {"code", "message", "request_id", "details"?}}`. Stable codes
  include `validation_failed`, `unauthorized`, `forbidden`, `read_only_key`, `read_only_replica`,
  `not_found`, `already_exists`, `conflict`, `id_conflict`, `dimension_mismatch`, `model_mismatch`,
  `quota_exceeded`, `rate_limited`, `overloaded`, `tenant_overloaded`, `busy`, `deadline_exceeded`,
  `graph_quarantined`, `replica_stale`, `encoder_failed`, `internal_error`. Internal errors are logged
  server-side with the same `request_id`; clients never see exception text, paths or stack traces.
* **Null and absent are the same.** The ISON text format is columnar and cannot tell a missing property
  from a `null` one (the library's own round trip turns one into the other), so the service never stores or
  returns null properties; sending `null` in a `PATCH` removes the property.

Not supported on purpose: undirected graphs (the file format cannot record directedness; use
`"direction": "both"`), property values that are lists or objects, and ids/types/property names outside the
whitelists above (they are exactly what survives the ISON round trip).

## Threat model

Assume: a hostile or buggy tenant holding a valid key (including a read-only one); anonymous network
callers; a tenant trying to infer other tenants' existence or data. Assume the operator and the host are
trusted, and that TLS is terminated in front of this process.

| Threat | Mitigation | Evidence |
| --- | --- | --- |
| Tenant B reads/writes/enumerates tenant A | The tenant is derived from the verified key only; a handler receives only that tenant's runtime; there is no global name-keyed map | `tests/test_isolation.py` - colliding ids and graph names, every endpoint incl. batch, search, subgraph, cursors; 3 deliberate code mutations each make it fail |
| Inference ("does graph X exist for someone else?") | A name that is not yours is simply absent: identical status, code and message to a name that exists nowhere; quota/usage errors report only your own numbers; metrics are aggregate with no tenant labels | same file |
| Forged / mixed credentials | Format `igv_<tenant>.<key-id>.<256-bit secret>`; the tenant in the key is only a lookup hint, authority is the HMAC check against that tenant's record; duplicate `Authorization` headers rejected | forgery, truncation, case, whitespace, duplicate-header and cross-tenant tests |
| Key theft from disk | Only `HMAC-SHA256(salt, secret)` and the salt are stored; `compare_digest`; unknown tenant/key id/revoked key still do one dummy HMAC; `tenant.json` is `0600` on POSIX | tests assert the secret is not in the file and is never logged |
| Brute force | Failed-auth throttle per peer address (`429`); 256-bit secrets | throttle test |
| Header/query/body tricks to select a tenant | None of them is read. Unknown query parameters and body fields are errors | `X-Tenant*`, `Host`, `Forwarded`, cookies, `?tenant=`, `?api_key=` tests |
| Path traversal / aliasing via names | Names are whitelist-matched (lowercase `[a-z0-9_-]`, so case-insensitive file systems cannot alias two names); the directory is `g-<name>` derived from the *validated* name and re-checked to be inside the base | 40 crafted names x GET/DELETE/POST/create, file tree compared before/after; traversal in replication routes |
| Request smuggling / framing | `Content-Length` only; chunked, duplicate or malformed lengths refused | `test_limits.py` raw-socket tests |
| Resource exhaustion | Body/header/URI caps; socket timeouts; bounded worker pool and accept queue (`503` when full); global and per-tenant concurrency gates; per-tenant token bucket; node/edge/graph/dimension/batch quotas; `top_k`/`max_hops` caps; response size cap; request deadlines; lock-wait timeouts | `test_limits.py` |
| Stack traces / secrets in responses and logs | Generic `internal_error` + request id; encoder errors are replaced by a generic message; log records pass a redaction filter and never include headers or bodies | tests |
| Corrupted or tampered snapshot | sha256 + size per file, library digest check, manifest-digest cross-check; fallback to the last good generation or quarantine | `test_persistence.py`, `test_replication.py` |
| Malicious replica source | A replica verifies everything before installing and refuses (keeps serving what it had). **It trusts the primary and the replication token.** | 6 tampering variants |

### What is NOT provided

Say these out loud before deploying:

* **No TLS.** The server speaks plain HTTP. Terminate TLS in a reverse proxy. API keys travel in a header.
* **Not hardened for the open internet.** It is a standard-library `socketserver` with sensible limits,
  not nginx. Slow-connection floods beyond the per-socket timeouts and the pool bound, HTTP/2, connection
  limits per address, and request-header memory (up to ~6 MB per connection in the worst case) are the
  proxy's job.
* **No remote admin API.** Tenants and keys are managed by the CLI on the data directory. Anyone with
  filesystem access to the data directory can mint keys.
* **No per-user permissions inside a tenant** beyond the `rw`/`ro` key role. No audit log beyond the
  request log. No key expiry (revoke manually). No encryption at rest.
* **Rate limiting is per tenant, in memory, per node.** Behind a router with several shards each tenant is
  on one shard, so its limit is exact; it is *not* shared across replicas.
* **A running computation cannot be interrupted.** Deadlines are checked between phases, and lock waits
  and encoder calls are bounded; a pathological traversal that is already running holds its worker until it
  finishes. It is bounded by the quotas (graph size, `max_hops`, `top_k`) and cannot stall other graphs.
* **Tenant existence is not a secret** (it is a prefix of the key, and registry lookups are slightly
  faster for known tenants). Key *validity* is.
* **The embedding endpoint sees every text.** With `encoder.kind = "http"` all node text and text queries
  go to that endpoint; with `mock` the "embeddings" are noise.
* **The metrics and replication tokens are single shared secrets** read from the environment
  (`IGV_METRICS_TOKEN`, `IGV_REPLICATION_TOKEN`, >= 24 characters). The replication token reads every
  tenant's data and key hashes: treat it like a database password.

## Persistence and durability

Per graph, under `<data>/tenants/<tenant>/graphs/g-<name>/`:

```
meta.json          immutable configuration
CURRENT            the generation to serve
gen-000007/        graph.isong  +  graph.isong.embeddings.json  +  MANIFEST.json (sha256, sizes, graph_digest)
```

A save builds the whole pair in a temp directory, re-parses it and checks it equals memory, fsyncs,
renames the directory into place, then atomically flips `CURRENT`. The library alone writes two separate
atomic files, so a crash between them leaves a mismatched pair (it detects that on load, but has nothing to
fall back to); generations fix that. On start every graph is verified (checksums, the library's
sidecar-vs-graph digest check, manifest digest). If `CURRENT`'s generation fails, the newest older one that
passes is served and the graph is reported `degraded`; if none passes the graph is **quarantined**
(`503 graph_quarantined`, files left untouched, other graphs unaffected). `strict_startup` makes any such
failure abort startup instead. `igv-server verify` does the same check offline.

* `durability = "snapshot"` (default): a write is acknowledged after it is applied in memory. A snapshot is
  written every `autosave_interval_seconds` (30 s), on `POST .../save`, and at graceful shutdown. **A crash
  (not a graceful stop) loses up to one autosave interval of acknowledged writes.** There is no write-ahead log.
* `durability = "sync"`: every mutating request writes a snapshot before it is acknowledged. No acknowledged
  write is lost (tested with `SIGKILL`), but each write costs O(graph size). Use it for small graphs.
* Snapshots hold the graph's read lock only while capturing the image (writers to *that graph* wait; readers
  do not): measured 0.20 s for 20,000 nodes with 64-dimensional vectors; the re-parse, checksums and fsync
  (a further 0.78 s there) run without any lock. Loading plus full verification of that graph takes 0.84 s.
* A data directory belongs to one process (an OS file lock enforces it).

## Scale-out and consistency

**Router** (`igv-server router --config ...`). Stateless; no credentials, no data. It reads the tenant id out
of the key, maps it to a shard with rendezvous hashing (or `router.pins`), and forwards the request unchanged.
The backend authenticates, so the router cannot widen access: a key naming tenant A but made for B is
forwarded to A's shard and fails there. Only `/v1/*` is proxied; `/metrics` and `/_repl` never are.
Reads (`GET`, and the five search `POST`s) may go to a replica if `read_preference = replica_preferred`
(fall back to the primary if the replica errors or reports `replica_stale`); a client can force the primary with
`X-Igv-Consistency: primary`. Writes always go to the primary and are **never retried**.

**Replicas** (`role = "replica"`, `replica.source = "dir:/path"` or `http://primary:8080` with
`IGV_REPLICATION_TOKEN`). A replica pulls saved generations and installs them only after verifying every
checksum and digest (see `replication.py`). Honest consistency statement:

1. A replica serves what the primary has *saved*. A read can be stale by up to
   `primary autosave interval + replica sync interval + sync time` (default 30 + 15 s plus transfer).
2. If the last fully successful sync is older than `replica.max_staleness_seconds` the replica answers
   `503 replica_stale` and reports not-ready, rather than serving unboundedly old data. Staleness survives a
   replica restart.
3. Graphs are synced independently: two graphs can reflect different primary moments. There is no
   cross-graph snapshot.
4. Read-your-writes is not provided on a replica. Revoking a key or disabling a tenant reaches replicas on
   the next sync, so a revoked key can keep working on a replica for up to the sync interval.
5. A snapshot that fails verification is refused and counted (`igv_replica_refused_total`); the replica keeps
   serving the previous generation, and the cycle is not counted as fresh, so repeated refusals eventually
   trip the staleness bound.

**Not provided:** automatic failover or promotion (a dead primary means its tenants get `502`/`503`); online
resharding or tenant migration (changing the shard list remaps tenants, but moving their data is manual: copy
the tenant directory to the new owner, or pin the tenant with `router.pins` meanwhile - `igv-server shard-of
<tenant> --config ...` tells you the owner); cross-tenant or cross-shard queries; streaming replication or a
write-ahead log; multi-primary writes. Capacity scales by *adding tenants' homes*, not by splitting one tenant
across nodes - one tenant's graphs live on one node and must fit its memory.

## Configuration

A JSON file (`--config` or `IGV_CONFIG`), see `config.example.json`, with environment overrides for the
common scalars (`IGV_DATA_DIR`, `IGV_HOST`, `IGV_PORT`, `IGV_ROLE`, `IGV_LOG_LEVEL`, `IGV_DURABILITY`,
`IGV_AUTOSAVE_INTERVAL_SECONDS`, `IGV_MAX_WORKERS`, `IGV_MAX_BODY_BYTES`, `IGV_STRICT_STARTUP`,
`IGV_REPLICA_SOURCE`). Secrets are environment-only and are *refused* in the file: `IGV_METRICS_TOKEN`,
`IGV_REPLICATION_TOKEN`, and the embedding endpoint's key (named by `encoder.api_key_env`). Unknown keys are
errors. Per-tenant quota overrides live in the tenant's `tenant.json` (`igv-server tenant create/set-quota
--quota name=value`) and take effect within `registry_recheck_seconds` (1 s).

## Operations

* **Logs.** JSON lines on stderr: `ts, level, msg`, and for requests `method, route (template), status, ms,
  tenant, request_id`. No headers, bodies, query text or ids. A redaction filter removes anything shaped like
  a key as a backstop.
* **Metrics.** `GET /metrics` (token): request counters by route and status class, a latency histogram per
  route, in-flight, auth failures/throttles, rate-limited, snapshots/failures, quarantined/degraded graphs,
  replica staleness/refusals. Aggregate only.
* **Shutdown.** `SIGTERM`/`SIGINT` (`CTRL_BREAK` on Windows): stop accepting, finish in-flight requests,
  close idle connections, write a final snapshot of every dirty graph, exit 0.
* **Backups.** Generations are immutable directories; copy `CURRENT` and the generation it names (or the
  whole data directory while the node is stopped) and `igv-server verify` the copy.
* **CLI.** `serve`, `router`, `tenant create|list|show|disable|enable|set-quota`, `key create|revoke`,
  `shard-of`, `verify`, `replica-sync` (one cycle, cron-friendly).

## Testing

```bash
cd isongraph-vector-server
pip install pytest numpy
python -m pytest tests            # ~1 minute; no network, no model download (MockEncoder)
```

147 tests (146 pass, 1 skipped on Windows: the POSIX file-mode check), about 70 s of test time on a quiet machine:

| file | tests | covers |
| --- | ---: | --- |
| `test_api_basic.py` | 16 | every endpoint's happy path, validation, error model, pagination, id rules |
| `test_isolation.py` | 30 | tenant isolation, credentials, crafted names, replication/metrics endpoints, log secrecy |
| `test_limits.py` | 23 | quotas (incl. a concurrent quota race), protocol limits over raw sockets, load shedding, deadlines, lock timeouts, prompt graceful stop, pipelining, byte-drip cut-off, `RWLock`/token bucket |
| `test_persistence.py` | 21 | snapshots, corruption fallback, quarantine, restart, `sync` durability, round-trip fuzzing, graceful shutdown |
| `test_crash.py` | 10 | real `SIGKILL` at five points inside a save, 10 random kills, `SIGKILL` of a whole server in `sync` mode, signal-driven shutdown |
| `test_replication.py` | 19 | dir and HTTP sources, incremental sync, six tampering variants refused, staleness bound, revocation, deletions |
| `test_router.py` | 13 | hashing properties, routing, forged keys, 502 without failover, writes never retried, replica reads and fallback |
| `test_lifecycle.py` | 15 | model change / `reembed`, traversal options, filters, ANN opt-in, CLI, config, logs, generic 500s |

See [Tested versus only claimed](#tested-versus-only-claimed) for exactly which properties are covered.

## Load test

`scripts/load_test.py` creates N tenants with M vectors each, starts the server (and optionally K shards and
a router) as separate processes, and drives closed-loop search load, reporting throughput and latency
percentiles as a table and JSON. Results from one development machine (Windows 11, 32 logical cores,
Python 3.12, MockEncoder, client-mode graphs, `top_k = 10`, vector queries) are in `scripts/results/`:

Single node, search throughput (`scripts/results/*.json`; "gen" = load-generator processes):

| setup | tenants x vectors | clients | req/s | p50 ms | p95 ms | p99 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 node, 1 generator | 8 x 2,000 (dim 64) | 1 | 1,609 | 0.57 | 0.75 | 1.2 |
| 1 node, 1 generator | 8 x 2,000 | 4 | 1,949 | 1.9 | 3.3 | 5.0 |
| 1 node, 1 generator | 8 x 2,000 | 16 | 2,000 | 6.9 | 16.4 | 22.1 |
| 1 node, 1 generator | 8 x 2,000 | 64 | 1,624 | 31.8 | 99.2 | 125.2 |
| 1 node, 1 generator | 32 x 2,000 | 16 | 1,895 | 7.7 | 15.5 | 20.7 |
| 1 node, 1 generator | 4 x 20,000 | 16 | 1,472 | 9.6 | 23.2 | 28.3 |
| 1 node, 1 generator, text queries | 8 x 2,000 | 16 | 1,945 | 6.5 | 19.3 | 24.9 |
| server inside the generator's process | 8 x 2,000 | 16 | 1,397 | 10.1 | 22.7 | 29.8 |

Scale-out (16 tenants x 2,000 vectors, 32 clients over 8 generator processes, nodes are separate processes):

| topology | req/s | p50 ms | p95 ms | p99 ms | vs. 1 node |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 node, direct | 2,102 | 12.5 | 34.6 | 49.5 | 1.00x |
| router -> 1 node | 1,818 | 14.9 | 38.1 | 57.9 | 0.86x (router overhead) |
| router -> 2 nodes | 2,977 | 8.4 | 28.8 | 42.1 | 1.42x |
| router -> 4 nodes | 3,333 | 9.2 | 15.4 | 19.1 | 1.59x |

Reading these honestly: a node tops out around 2,000 small searches per second (one Python process, GIL).
Adding shards helps but sub-linearly here, because the router is itself a single Python process, the
generators and four servers shared one 32-thread desktop, and every request pays an extra hop. If the router
becomes the limit, run several routers (they are stateless). Zero errors in every run.

These are *this machine, this workload* numbers for a GIL-bound pure-Python server; they say nothing about
other hardware, and the load generator shares the machine with the server. Use the script on your own.

## Known limits

* Throughput per node is bounded by the GIL (a few thousand small searches per second); scale by adding
  shard processes, not threads.
* All of a node's graphs stay in memory (loaded at startup, never evicted). Memory is bounded only by the
  quotas you set.
* A snapshot rewrites the whole graph; very large graphs make `sync` durability and snapshot lock time
  expensive. The library targets small-to-medium graphs, and so does this.
* `reembed` holds the graph's write lock while it calls the encoder.
* The ISON file format forces: directed graphs only; scalar property values; no null/absent distinction;
  no unicode whitespace at value edges (rejected up front).
* The standard-library server is single-process, HTTP/1.1, no TLS.

## Tested versus only claimed

**Exercised by a test that fails if the property breaks** (three deliberate mutations of the source - a
shared tenant runtime, a loosened graph-name regex, a key lookup that ignores the tenant - each made the
isolation suite fail, then were reverted):

* tenant isolation on every endpoint including batch, search, subgraph, cursors, with identical ids and graph
  names in several tenants, concurrently; uniform not-found for foreign and absent names; own-only usage in
  quota errors, listings and usage; read-only keys; revocation and disabling
* key forgery / truncation / case / duplicate-header variants; key never in file, response or log
* 40+ crafted names and ids against path traversal (file tree compared before and after); traversal in the
  replication file routes; the replication and metrics tokens are not tenant keys and vice versa
* body/header/URI limits, chunked and conflicting `Content-Length`, slow clients, pool exhaustion, per-tenant
  and global load shedding, deadlines, lock timeouts, exact and race-free quotas
* atomic snapshots under `SIGKILL` at each step, random kills, `SIGKILL` of the real server with `sync`
  durability, corruption fallback, quarantine isolation, round-trip fidelity of everything validation admits
* replica refusal on six distinct tampering patterns including a manifest whose checksums were re-written to
  match; staleness cut-off; deletions; "unreadable is not deleted"
* router: forged tenant prefix, no failover, no write retry, replica fallback, the hash's balance and
  minimal-disruption properties

**Claimed but not tested here:** behaviour under TLS or behind a particular proxy; HTTP encoder end to end (the
library's `HttpEncoder` is used as-is; this repo only tests encoder *failures*); a real
sentence-transformers model; POSIX-only paths (`fsync` of directories, `0600` mode - the mode check is
skipped on Windows, SIGTERM handling is covered via `CTRL_BREAK` on Windows); the Dockerfile (no Docker
daemon was available, so it was not built; the two `pip install --no-deps` steps it runs were checked
separately); timing-side-channel resistance (the code takes the same path for unknown tenant, unknown key id
and wrong secret and the responses are byte-identical, but no timing measurement was made); memory limits;
long soak runs; multi-day replica drift.
