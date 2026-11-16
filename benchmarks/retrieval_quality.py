#!/usr/bin/env python3
"""
Retrieval-quality benchmark for isongraph-vector.

Question answered: when the answer to a question is spread over nodes that are
connected by RELATIONSHIPS rather than by wording, does graph-aware retrieval
(semantic_multi_hop / semantic_subgraph) pull in more of the needed context
than a plain vector top-k - and what does it cost in extra nodes?

The data is a synthetic, seeded "operations knowledge graph":

    incident -AFFECTS->    service
    incident -RELATED_TO-> incident          (distractor edges)
    service  -DEPENDS_ON-> service | datastore
    service  -OWNED_BY->   team
    service  -HAS_RUNBOOK->runbook
    datastore-OWNED_BY->   team
    team     -ONCALL->     person
    team     -HAS_MEMBER-> person            (distractor edges)

and four question types, each with ground truth computed from the graph
structure (not from the retrieval methods):

    incident_owner      symptoms -> service -> team -> on-call person   (3 hops)
    incident_rootcause  symptoms -> service -> dependencies -> their teams (3 hops)
    service_deps        service name -> its team, dependencies, and their dependencies (2 hops)
    impact              datastore name -> the services that depend on it (1 hop, INCOMING edges)

For every question there are two gold sets:

    seed_gold     nodes the question text directly describes (the incident or
                  the named service/datastore); a vector search *should* find
                  these
    support_gold  nodes needed to answer that are reachable only through
                  relationships; their own text generally does not resemble
                  the question

Embeddings need no model download: ``TopicHashEncoder`` is a deterministic
hashed bag-of-words/bigram encoder with a small synonym lexicon so paraphrases
and topic clusters overlap the way real embeddings do, only more crudely.
``--encoder st`` swaps in sentence-transformers (all-MiniLM-L6-v2) if installed.

Usage:
    python benchmarks/retrieval_quality.py                 # default run
    python benchmarks/retrieval_quality.py --quick         # smaller, seconds
    python benchmarks/retrieval_quality.py --encoder st    # real embeddings
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import environment_info, import_package  # noqa: E402

ivec = import_package()
from ison_graph import Direction  # noqa: E402

SemanticGraph = ivec.SemanticGraph
EmbeddingEncoder = ivec.EmbeddingEncoder

NodeRef = Tuple[str, str]

# =============================================================================
# Vocabulary
# =============================================================================

AREAS: Dict[str, List[str]] = {
    "commerce": ["payments", "checkout", "billing", "invoicing", "cart", "pricing", "orders", "refunds"],
    "data": ["analytics", "reporting", "warehouse", "etl", "metrics", "search", "recommendations", "catalog"],
    "identity": ["auth", "sso", "profile", "sessions", "permissions", "accounts", "audit", "consent"],
    "logistics": ["shipping", "inventory", "fulfillment", "returns", "routing", "tracking", "suppliers", "depots"],
    "comms": ["notifications", "email", "sms", "chat", "push", "webhooks", "templates", "campaigns"],
}
APP_KINDS = ["api", "gateway", "worker", "scheduler", "web", "service", "ingest", "sync"]
LOCAL_STORES = ["db", "cache", "queue"]
STORE_TECH = {"db": "PostgreSQL database", "cache": "Redis cache", "queue": "message queue"}
SHARED_TECH = ["pg-core", "redis-shared", "kafka-events", "mysql-legacy", "s3-blob", "elastic-logs"]

# symptom concept -> synonym words (the encoder collapses these onto one concept)
SYMPTOMS: Dict[str, List[str]] = {
    "latency": ["latency", "slow", "slowness", "timeouts", "sluggish", "delays", "lag", "stalls"],
    "errors": ["errors", "failures", "exceptions", "crashes", "5xx", "faults", "broken", "failing"],
    "capacity": ["saturation", "overload", "exhausted", "throttled", "backlog", "spike", "congestion", "queueing"],
    "data": ["corrupt", "stale", "inconsistent", "missing", "duplicated", "lost", "mismatch", "drift"],
    "access": ["denied", "unauthorized", "forbidden", "lockout", "expired", "rejected", "blocked", "revoked"],
    "outage": ["outage", "down", "unreachable", "unavailable", "offline", "dead", "unresponsive", "blackout"],
    "deploy": ["rollout", "release", "deploy", "regression", "rollback", "migration", "upgrade", "patch"],
    "cost": ["bill", "spend", "overrun", "budget", "quota", "metered", "expensive", "surge"],
}
FILLER = ["users", "reported", "seeing", "after", "during", "intermittent", "sustained", "across", "several", "regions", "customers", "noticed"]

STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "is", "are", "for", "it", "its", "by", "as", "at", "be",
        "this", "that", "with", "from", "which", "what", "who", "if", "does", "do", "can", "has", "have", "their", "them"}

FIRST = ["Priya", "Marcus", "Elena", "Tomas", "Aisha", "Kenji", "Sofia", "Diego", "Mei", "Oskar", "Lena", "Rahul",
         "Chloe", "Ivan", "Nadia", "Felix", "Hana", "Jonas", "Amara", "Leo"]
LAST = ["Nair", "Chen", "Okafor", "Silva", "Novak", "Haddad", "Kim", "Rossi", "Larsen", "Mehta", "Duarte", "Weber"]
TEAM_SYL = ["Atlas", "Borealis", "Cobalt", "Dynamo", "Ember", "Falcon", "Granite", "Helix", "Ion", "Juno", "Keystone",
            "Lumen", "Meridian", "Nimbus", "Orion", "Pulsar", "Quartz", "Rift", "Sable", "Tundra", "Umbra", "Vertex"]


# =============================================================================
# Encoders
# =============================================================================

def _tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    out: List[str] = []
    for t in toks:
        if t in STOP:
            continue
        out.append(t)
        if "-" in t:  # name "payments-api" matches both itself and "payments"
            out.extend(p for p in t.split("-") if p and p not in STOP)
    return out


class TopicHashEncoder(EmbeddingEncoder):
    """
    Deterministic text -> vector with controlled semantic overlap, no model.

    Features are hashed (signed, crc32) into ``dimension`` buckets:
      * every word            weight 0.7   (exact lexical overlap)
      * adjacent word pairs   weight 0.4   (phrase overlap)
      * symptom synonyms      weight 1.0 on a shared concept token
                              (paraphrases overlap without sharing a word)
      * domain -> area        weight 0.5 on a shared area token
                              (payments and billing are "commerce": topic clusters)
    then the vector is L2-normalised. With 256 buckets, unrelated texts collide
    a little, as in a real embedding space. It is crude on purpose: it is a
    control for the benchmark, not a claim about any production model.
    """

    model_name = "topic-hash-256"

    def __init__(self, dimension: int = 256):
        self._dimension = dimension
        self.model_name = f"topic-hash-{dimension}"
        self._word_concept: Dict[str, str] = {}
        for concept, words in SYMPTOMS.items():
            for w in words:
                self._word_concept[w] = "sym:" + concept
        self._word_area: Dict[str, str] = {}
        for area, domains in AREAS.items():
            for d in domains:
                self._word_area[d] = "area:" + area

    @property
    def dimension(self) -> int:
        return self._dimension

    def _add(self, vec: List[float], feature: str, weight: float) -> None:
        h = zlib.crc32(feature.encode("utf-8"))
        vec[h % self._dimension] += weight if (h >> 16) & 1 else -weight

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self._dimension
        toks = _tokens(text)
        for t in toks:
            self._add(vec, "w:" + t, 0.7)
            c = self._word_concept.get(t)
            if c:
                self._add(vec, c, 1.0)
            a = self._word_area.get(t)
            if a:
                self._add(vec, a, 0.5)
        for a, b in zip(toks, toks[1:]):
            self._add(vec, "b:" + a + "|" + b, 0.4)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]


def make_encoder(kind: str):
    if kind == "hash":
        return TopicHashEncoder(256)
    if kind == "st":
        try:
            return ivec.SentenceTransformerEncoder()
        except Exception as exc:  # ImportError, offline model download, ...
            raise SystemExit(f"--encoder st unavailable: {type(exc).__name__}: {exc}")
    raise SystemExit(f"unknown encoder {kind!r}")


# =============================================================================
# Synthetic knowledge graph + questions
# =============================================================================

class Dataset:
    def __init__(self):
        self.nodes: Dict[NodeRef, dict] = {}            # ref -> {"text":..., "props":...}
        self.edges: List[Tuple[str, NodeRef, NodeRef]] = []
        self.out: Dict[NodeRef, List[Tuple[str, NodeRef]]] = defaultdict(list)
        self.inc: Dict[NodeRef, List[Tuple[str, NodeRef]]] = defaultdict(list)
        self.questions: List[dict] = []

    def add_node(self, ntype: str, nid: str, text: str, **props) -> NodeRef:
        ref = (ntype, nid)
        self.nodes[ref] = {"text": text, "props": dict(props, name=props.get("name", nid))}
        return ref

    def add_edge(self, rel: str, src: NodeRef, dst: NodeRef) -> None:
        self.edges.append((rel, src, dst))
        self.out[src].append((rel, dst))
        self.inc[dst].append((rel, src))

    def targets(self, ref: NodeRef, rel: str) -> List[NodeRef]:
        return [d for r, d in self.out[ref] if r == rel]

    def sources(self, ref: NodeRef, rel: str) -> List[NodeRef]:
        return [s for r, s in self.inc[ref] if r == rel]


def build_dataset(seed: int, n_apps: int, incidents_per_app: float) -> Dataset:
    rng = random.Random(seed)
    ds = Dataset()

    # ---- teams, people ------------------------------------------------------
    domains = [(area, d) for area, ds_ in AREAS.items() for d in ds_]
    team_names = TEAM_SYL[:]
    rng.shuffle(team_names)
    teams: Dict[str, NodeRef] = {}
    domain_team: Dict[str, NodeRef] = {}
    oncall: Dict[NodeRef, NodeRef] = {}
    pidx = 0

    def new_person(team_ref: NodeRef, team_label: str, primary: bool) -> NodeRef:
        nonlocal pidx
        pidx += 1
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if primary:
            text = f"{name}, on-call engineer (primary pager) for team {team_label}"
        else:
            text = f"{name}, engineer, member of team {team_label}"
        return ds.add_node("person", f"p{pidx:04d}", text, name=name)

    def new_team(label: str, scope: str) -> NodeRef:
        ref = ds.add_node("team", label.lower(),
                          f"Team {label}, engineering group responsible for {scope}. Owns services and runs the pager rotation.",
                          name=f"Team {label}")
        p = new_person(ref, label, True)
        ds.add_edge("ONCALL", ref, p)
        ds.add_edge("HAS_MEMBER", ref, p)
        for _ in range(3):
            ds.add_edge("HAS_MEMBER", ref, new_person(ref, label, False))
        oncall[ref] = p
        return ref

    # one team per two domains; each team's blurb names its AREA only, not the services
    for i in range(0, len(domains), 2):
        label = team_names[(i // 2) % len(team_names)] + ("" if i // 2 < len(team_names) else str(i))
        area = domains[i][0]
        t = new_team(label, f"the {area} platform")
        for _, d in domains[i:i + 2]:
            domain_team[d] = t
    platform_teams = [new_team("Foundation", "shared infrastructure and storage"),
                      new_team("Bedrock", "shared infrastructure and data platform")]

    # ---- datastores ----------------------------------------------------------
    local_store: Dict[Tuple[str, str], NodeRef] = {}
    for _, d in domains:
        for kind in LOCAL_STORES:
            name = f"{d}-{kind}"
            ref = ds.add_node("datastore", name, f"{name}: {STORE_TECH[kind]} holding {d} data.", name=name)
            ds.add_edge("OWNED_BY", ref, domain_team[d])
            local_store[(d, kind)] = ref
    n_shared = max(12, n_apps // 6)
    shared: List[NodeRef] = []
    for i in range(n_shared):
        tech = SHARED_TECH[i % len(SHARED_TECH)]
        name = f"{tech}-{chr(97 + i // len(SHARED_TECH))}{i % 3}"
        ref = ds.add_node("datastore", name, f"{name}: shared {tech.split('-')[0]} cluster used by many teams, tier 0.", name=name)
        ds.add_edge("OWNED_BY", ref, rng.choice(platform_teams))
        shared.append(ref)

    # ---- apps ----------------------------------------------------------------
    all_names = [(area, d, k) for area, d in domains for k in APP_KINDS]
    rng.shuffle(all_names)
    chosen = all_names[:n_apps]
    apps: List[NodeRef] = []
    app_area: Dict[NodeRef, str] = {}
    app_domain: Dict[NodeRef, str] = {}
    for area, d, k in chosen:
        name = f"{d}-{k}"
        ref = ds.add_node("service", name, f"Service {name}: {k} component of the {d} domain in the {area} platform, tier {rng.randint(1, 3)}.", name=name)
        apps.append(ref)
        app_area[ref] = area
        app_domain[ref] = d
        ds.add_edge("OWNED_BY", ref, domain_team[d])
        rb = ds.add_node("runbook", f"rb-{name}", f"Runbook for {name}: restart procedure, rollback steps, escalation contacts.", name=f"Runbook {name}")
        ds.add_edge("HAS_RUNBOOK", ref, rb)

    for i, app in enumerate(apps):
        d = app_domain[app]
        deps = [local_store[(d, "db")]]
        if rng.random() < 0.5:
            deps.append(local_store[(d, rng.choice(["cache", "queue"]))])
        if rng.random() < 0.4:
            deps.append(rng.choice(shared))
        earlier = [a for a in apps[:i] if a != app]
        k_apps = rng.choice([0, 0, 1, 1, 2])
        same_area = [a for a in earlier if app_area[a] == app_area[app]]
        for _ in range(k_apps):
            pool = same_area if (same_area and rng.random() < 0.6) else earlier
            if pool:
                deps.append(rng.choice(pool))
        deps = list(dict.fromkeys(deps))
        # cap at 4 dependencies, dropping app-to-app edges first
        while len(deps) > 4:
            for j in range(len(deps) - 1, -1, -1):
                if deps[j][0] == "service":
                    deps.pop(j)
                    break
            else:
                deps.pop()
        for dep in deps:
            ds.add_edge("DEPENDS_ON", app, dep)

    # ---- incidents -----------------------------------------------------------
    n_inc = int(n_apps * incidents_per_app)
    concepts = list(SYMPTOMS)
    incidents: List[Tuple[NodeRef, NodeRef, str, Optional[str], str, str]] = []
    for i in range(n_inc):
        app = rng.choice(apps)
        concept = rng.choice(concepts)
        phrase = symptom_phrase(rng, concept)
        d = app_domain[app]
        parts = [f"INC-{i:04d}: {phrase}."]
        mention_domain = rng.random() < 0.6
        if mention_domain:
            parts.append(f"Affects the {d} flow.")
        if rng.random() < 0.3:
            parts.append(f"Alert fired on {ds.nodes[app]['props']['name']}.")
        code = f"e{rng.randint(1000, 9999)}"
        parts.append(f"Reference code {code}.")
        ref = ds.add_node("incident", f"INC-{i:04d}", " ".join(parts), name=f"INC-{i:04d}")
        ds.add_edge("AFFECTS", ref, app)
        incidents.append((ref, app, concept, d if mention_domain else None, code, phrase))
    for k, (ref, app, *_rest) in enumerate(incidents):
        if k > 0 and rng.random() < 0.3:
            ds.add_edge("RELATED_TO", ref, incidents[rng.randrange(k)][0])

    # ---- questions -----------------------------------------------------------
    ds._incidents = incidents  # type: ignore[attr-defined]
    ds._apps = apps            # type: ignore[attr-defined]
    ds._oncall = oncall        # type: ignore[attr-defined]
    return ds


def symptom_phrase(rng: random.Random, concept: str) -> str:
    words = rng.sample(SYMPTOMS[concept], 3)
    fill = rng.sample(FILLER, 2)
    return f"{fill[0]} {words[0]} and {words[1]} {fill[1]} {words[2]}"


def make_questions(ds: Dataset, n_questions: int, max_gold: int, seed: int) -> List[dict]:
    rng = random.Random(seed + 1)
    per_type = max(1, n_questions // 4)
    incidents = ds._incidents  # type: ignore[attr-defined]
    apps = ds._apps            # type: ignore[attr-defined]
    oncall = ds._oncall        # type: ignore[attr-defined]
    questions: List[dict] = []

    def incident_query(inc, tail: str) -> str:
        ref, app, concept, domain, code, phrase = inc
        parts = [symptom_phrase(rng, concept) + "."]  # a *different* paraphrase of the same symptom
        has_domain = domain is not None and rng.random() < 0.8
        has_code = rng.random() < 0.7 or not has_domain
        if has_domain:
            parts.append(f"It hits the {domain} flow.")
        if has_code:
            parts.append(f"Reference code {code}.")
        parts.append(tail)
        return " ".join(parts)

    def add(qtype, text, seed_gold, support_gold, direction):
        sg, sp = set(seed_gold), set(support_gold) - set(seed_gold)
        if len(sg | sp) > max_gold or not sp:
            return False
        questions.append({"id": f"{qtype}-{len(questions):03d}", "type": qtype, "text": text,
                          "seed_gold": sorted(sg), "support_gold": sorted(sp), "natural_direction": direction})
        return True

    def tries(qtype, gen):
        got, attempts = 0, 0
        while got < per_type and attempts < per_type * 60:
            attempts += 1
            got += bool(gen())

    def q_owner():
        inc = rng.choice(incidents)
        svc = ds.targets(inc[0], "AFFECTS")[0]
        team = ds.targets(svc, "OWNED_BY")[0]
        text = incident_query(inc, "Who owns the affected service and who is on call for it?")
        return add("incident_owner", text, [inc[0]], [svc, team, oncall[team]], "out")

    def q_root():
        inc = rng.choice(incidents)
        svc = ds.targets(inc[0], "AFFECTS")[0]
        deps = ds.targets(svc, "DEPENDS_ON")
        dep_teams = {t for d in deps for t in ds.targets(d, "OWNED_BY")}
        text = incident_query(inc, "Which upstream dependencies of the affected service could be responsible, and which teams own them?")
        return add("incident_rootcause", text, [inc[0]], [svc, *deps, *dep_teams], "out")

    def q_deps():
        svc = rng.choice(apps)
        deps = ds.targets(svc, "DEPENDS_ON")
        deps2 = {x for d in deps for x in ds.targets(d, "DEPENDS_ON")}
        team = ds.targets(svc, "OWNED_BY")
        name = ds.nodes[svc]["props"]["name"]
        text = f"What does {name} depend on, including indirect dependencies, and which team owns it?"
        return add("service_deps", text, [svc], [*deps, *deps2, *team], "out")

    stores = [r for r in ds.nodes if r[0] == "datastore"]

    def q_impact():
        st = rng.choice(stores)
        dependents = ds.sources(st, "DEPENDS_ON")
        name = ds.nodes[st]["props"]["name"]
        text = f"Which services would be affected if {name} goes down?"
        return add("impact", text, [st], dependents, "in")

    tries("incident_owner", q_owner)
    tries("incident_rootcause", q_root)
    tries("service_deps", q_deps)
    tries("impact", q_impact)
    return questions


# =============================================================================
# Retrieval variants
# =============================================================================

DIRS = {"out": Direction.OUT, "in": Direction.IN, "both": Direction.BOTH}

# name -> dict(method params). "vector_only" is special-cased.
VARIANTS: List[Tuple[str, dict]] = [
    ("vector_only", {}),
    ("hop2_out_decay0.8", dict(max_hops=2, direction="out", decay=0.8)),
    ("hop3_out_decay0.8", dict(max_hops=3, direction="out", decay=0.8)),
    ("hop3_out_decay0.5", dict(max_hops=3, direction="out", decay=0.5)),
    ("hop3_out_decay0.95", dict(max_hops=3, direction="out", decay=0.95)),
    ("hop3_out_blend0.3", dict(max_hops=3, direction="out", decay=0.8, blend=0.3)),
    ("hop3_out_seeds1", dict(max_hops=3, direction="out", decay=0.8, top_k_seeds=1)),
    ("hop3_out_seeds2", dict(max_hops=3, direction="out", decay=0.8, top_k_seeds=2)),
    ("hop2_in_decay0.8", dict(max_hops=2, direction="in", decay=0.8)),
    ("hop3_both_decay0.8", dict(max_hops=3, direction="both", decay=0.8)),
    ("hop3_both_blend0.3", dict(max_hops=3, direction="both", decay=0.8, blend=0.3)),
]


def build_graph(ds: Dataset, encoder) -> SemanticGraph:
    g = SemanticGraph(name="ops", embedding_db=":memory:", encoder=encoder, auto_embed=False)
    for (ntype, nid), info in ds.nodes.items():
        g.add_node(ntype, nid, **info["props"])
    for rel, src, dst in ds.edges:
        g.add_edge(rel, src, dst)
    g.embed_nodes([(ref, info["text"]) for ref, info in ds.nodes.items()])
    return g


def undirected_adj(ds: Dataset) -> Dict[NodeRef, Set[NodeRef]]:
    adj: Dict[NodeRef, Set[NodeRef]] = defaultdict(set)
    for _, s, d in ds.edges:
        adj[s].add(d)
        adj[d].add(s)
    return adj


def count_components(nodes: Set[NodeRef], adj) -> int:
    seen: Set[NodeRef] = set()
    comps = 0
    for n in nodes:
        if n in seen:
            continue
        comps += 1
        stack = [n]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(x for x in adj[cur] if x in nodes and x not in seen)
    return comps


def run_one(graph: SemanticGraph, method: str, name: str, params: dict, text: str, k: int,
            threshold: float) -> Tuple[Set[NodeRef], Optional[int], float]:
    """Return (node set, components-from-library-or-None, seconds)."""
    t0 = time.perf_counter()
    if name == "vector_only":
        res = graph.similarity_search(text, top_k=k)
        nodes = {r.node_ref for r in res}
        return nodes, None, time.perf_counter() - t0
    kw = dict(rel_type=None, max_hops=params["max_hops"], top_k_seeds=params.get("top_k_seeds", 5),
              top_k_results=k, direction=DIRS[params["direction"]], decay=params["decay"],
              threshold=threshold, blend=params.get("blend", 0.0))
    if method == "multi_hop":
        res = graph.semantic_multi_hop(text, **kw)
        nodes = {r.node_ref for r in res}
        return nodes, None, time.perf_counter() - t0
    sub = graph.semantic_subgraph(text, **kw)
    nodes = {n.ref for n in sub.nodes()}
    comps = len(sub.connected_components())
    sub.close()
    return nodes, comps, time.perf_counter() - t0


def evaluate(graph, ds, questions, budgets, threshold, adj, progress=True):
    # records[(variant, method, k)] -> list of per-question dicts
    records: Dict[Tuple[str, str, int], List[dict]] = defaultdict(list)
    mismatches = 0
    comp_mismatches = 0
    total = sum(1 if n == "vector_only" else 2 for n, _ in VARIANTS) * len(budgets) * len(questions)
    done = 0
    for name, params in VARIANTS:
        methods = ["vector"] if name == "vector_only" else ["multi_hop", "subgraph"]
        for k in budgets:
            for method in methods:
                for q in questions:
                    nodes, lib_comps, dt = run_one(graph, method, name, params, q["text"], k, threshold)
                    seed_g, sup_g = set(map(tuple, q["seed_gold"])), set(map(tuple, q["support_gold"]))
                    gold = seed_g | sup_g
                    own_comps = count_components(nodes, adj)
                    if lib_comps is not None and lib_comps != own_comps:
                        comp_mismatches += 1
                    records[(name, method, k)].append({
                        "qid": q["id"], "type": q["type"],
                        "seed_recall": len(nodes & seed_g) / len(seed_g),
                        "support_recall": len(nodes & sup_g) / len(sup_g),
                        "context_recall": len(nodes & gold) / len(gold),
                        "precision": (len(nodes & gold) / len(nodes)) if nodes else 0.0,
                        "complete": float(gold <= nodes),
                        "nodes": len(nodes),
                        "components": own_comps,
                        "ms": dt * 1000.0,
                        "_nodes": nodes,
                    })
                done += len(questions)
        if progress:
            print(f"  [{done}/{total}] {name}", file=sys.stderr, flush=True)
    # sanity: subgraph must return the same node set as multi_hop
    for (name, method, k), recs in records.items():
        if method != "subgraph":
            continue
        for a, b in zip(recs, records[(name, "multi_hop", k)]):
            if a["_nodes"] != b["_nodes"]:
                mismatches += 1
    return records, mismatches, comp_mismatches


# =============================================================================
# Aggregation / reporting
# =============================================================================

METRICS = ["seed_recall", "support_recall", "context_recall", "precision", "complete", "nodes", "components", "ms"]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def ci95(xs):
    if len(xs) < 2:
        return 0.0
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def aggregate(recs: List[dict]) -> dict:
    out = {m: mean([r[m] for r in recs]) for m in METRICS}
    out["context_recall_ci95"] = ci95([r["context_recall"] for r in recs])
    out["ms_median"] = statistics.median([r["ms"] for r in recs])
    out["n"] = len(recs)
    return out


def wtl(recs, base_recs, key="context_recall", eps=1e-9):
    w = t = l = 0
    for a, b in zip(recs, base_recs):
        if a[key] > b[key] + eps:
            w += 1
        elif a[key] < b[key] - eps:
            l += 1
        else:
            t += 1
    return f"{w}/{t}/{l}"


def make_report(ds, questions, records, budgets, args, mismatches, comp_mismatches, graph):
    qtypes = sorted({q["type"] for q in questions})
    result = {
        "config": vars(args),
        "environment": environment_info(),
        "graph": {"nodes": len(ds.nodes), "edges": len(ds.edges),
                  "by_type": {t: sum(1 for r in ds.nodes if r[0] == t) for t in sorted({r[0] for r in ds.nodes})}},
        "questions": {"total": len(questions),
                      "by_type": {t: sum(1 for q in questions if q["type"] == t) for t in qtypes},
                      "mean_gold_size": mean([len(q["seed_gold"]) + len(q["support_gold"]) for q in questions]),
                      "mean_support_size": mean([len(q["support_gold"]) for q in questions])},
        "sanity": {"subgraph_vs_multi_hop_node_set_mismatches": mismatches,
                   "component_count_mismatches_vs_library": comp_mismatches},
        "results": {},
    }
    md: List[str] = []
    md.append("# Retrieval quality benchmark - sample output\n")
    md.append(f"Encoder: `{args.encoder}` | seed: {args.seed} | graph: {result['graph']['nodes']} nodes, "
              f"{result['graph']['edges']} edges | questions: {len(questions)} "
              f"({', '.join(f'{k}={v}' for k, v in result['questions']['by_type'].items())}) | "
              f"mean gold size {result['questions']['mean_gold_size']:.1f} nodes "
              f"({result['questions']['mean_support_size']:.1f} supporting)\n")
    md.append("Seed threshold: %.2f. Budget k = max nodes returned (vector top-k, or top_k_results). "
              "Sanity: semantic_subgraph node set == semantic_multi_hop node set in all cases "
              "(mismatches: %d); component counts agree with the library's own `connected_components()` "
              "(mismatches: %d).\n" % (args.threshold, mismatches, comp_mismatches))

    for k in budgets:
        base = records[("vector_only", "vector", k)]
        # Questions whose seed nodes plain vector top-k already finds: a fixed
        # subset on which "can the graph step reach the supporting nodes?" is
        # separated from "could the question be matched at all?".
        hit_ids = {r["qid"] for r in base if r["seed_recall"] == 1.0}
        md.append(f"\n## Budget k = {k}\n")
        md.append(f"`support|seed-hit` = support recall restricted to the {len(hit_ids)} of {len(base)} questions "
                  f"whose seed nodes plain vector top-{k} already finds (same subset for every row).\n")
        md.append("| variant | method | seed recall | support recall | support\\|seed-hit | context recall (+-95% CI) | complete | precision | nodes | components | W/T/L vs vector | median ms |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        result["results"][str(k)] = {}
        for name, _ in VARIANTS:
            for method in (["vector"] if name == "vector_only" else ["multi_hop", "subgraph"]):
                recs = records[(name, method, k)]
                a = aggregate(recs)
                a["wtl_vs_vector"] = wtl(recs, base) if name != "vector_only" else None
                a["support_recall_seed_hit_subset"] = mean([r["support_recall"] for r in recs if r["qid"] in hit_ids])
                a["by_type"] = {t: aggregate([r for r in recs if r["type"] == t]) for t in qtypes}
                result["results"][str(k)][f"{name}/{method}"] = a
                md.append(f"| {name} | {method} | {a['seed_recall']:.3f} | {a['support_recall']:.3f} | "
                          f"{a['support_recall_seed_hit_subset']:.3f} | {a['context_recall']:.3f} +-{a['context_recall_ci95']:.3f} | {a['complete']:.2f} | "
                          f"{a['precision']:.3f} | {a['nodes']:.1f} | {a['components']:.2f} | "
                          f"{a['wtl_vs_vector'] or '-'} | {a['ms_median']:.2f} |")

        md.append(f"\n### Context recall by question type (k = {k}; multi_hop rows; subgraph returns the same nodes)\n")
        md.append("| variant | " + " | ".join(qtypes) + " |")
        md.append("|---|" + "---|" * len(qtypes))
        for name, _ in VARIANTS:
            method = "vector" if name == "vector_only" else "multi_hop"
            r = result["results"][str(k)][f"{name}/{method}"]["by_type"]
            md.append(f"| {name} | " + " | ".join(f"{r[t]['context_recall']:.3f}" for t in qtypes) + " |")

        md.append(f"\n### Seed recall vs support recall by type (k = {k}) - vector_only vs best graph variant\n")
        graph_rows = [(n, result["results"][str(k)][f"{n}/multi_hop"]) for n, _ in VARIANTS if n != "vector_only"]
        best_name, best = max(graph_rows, key=lambda x: x[1]["context_recall"])
        vb = result["results"][str(k)]["vector_only/vector"]
        md.append(f"Best graph variant by overall context recall: **{best_name}**.\n")
        md.append("| type | vector seed | vector support | best seed | best support |")
        md.append("|---|---|---|---|---|")
        for t in qtypes:
            md.append(f"| {t} | {vb['by_type'][t]['seed_recall']:.3f} | {vb['by_type'][t]['support_recall']:.3f} | "
                      f"{best['by_type'][t]['seed_recall']:.3f} | {best['by_type'][t]['support_recall']:.3f} |")
        result["results"][str(k)]["_best_graph_variant"] = best_name
    return result, "\n".join(md) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--encoder", choices=["hash", "st"], default="hash",
                    help="hash: deterministic offline topic encoder (default); st: sentence-transformers all-MiniLM-L6-v2")
    ap.add_argument("--services", type=int, default=240, help="number of application services")
    ap.add_argument("--incidents-per-service", type=float, default=4.0)
    ap.add_argument("--questions", type=int, default=200, help="total, split evenly over 4 question types")
    ap.add_argument("--budgets", default="10,20", help="comma list of node budgets k")
    ap.add_argument("--threshold", type=float, default=0.1, help="minimum seed similarity")
    ap.add_argument("--quick", action="store_true", help="small graph and 40 questions (seconds)")
    ap.add_argument("--out-prefix", default=str(Path(__file__).resolve().parent / "results" / "retrieval_quality"),
                    help="writes <prefix>.md and <prefix>.json; pass '' to skip")
    args = ap.parse_args(argv)
    if args.quick:
        args.services, args.questions = 80, 40
    budgets = sorted(int(x) for x in args.budgets.split(","))

    t0 = time.time()
    encoder = make_encoder(args.encoder)
    ds = build_dataset(args.seed, args.services, args.incidents_per_service)
    questions = make_questions(ds, args.questions, max_gold=budgets[0], seed=args.seed)
    print(f"graph: {len(ds.nodes)} nodes, {len(ds.edges)} edges; {len(questions)} questions "
          f"(gold <= {budgets[0]} nodes so the smallest budget can hold it)", file=sys.stderr)
    graph = build_graph(ds, encoder)
    print(f"embedded in {time.time() - t0:.1f}s; evaluating...", file=sys.stderr)

    adj = undirected_adj(ds)
    records, mism, cmism = evaluate(graph, ds, questions, budgets, args.threshold, adj)
    result, md = make_report(ds, questions, records, budgets, args, mism, cmism, graph)
    result["questions_sample"] = questions[:8]
    result["runtime_s"] = round(time.time() - t0, 1)
    print(md)
    if args.out_prefix:
        prefix = Path(args.out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".md").write_text(md, encoding="utf-8")
        prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"wrote {prefix.with_suffix('.md')} and {prefix.with_suffix('.json')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
