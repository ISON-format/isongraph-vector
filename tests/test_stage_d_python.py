#!/usr/bin/env python3
"""
Stage D tests for the Python port: one copy of every vector, the numpy batch
search, and the optional binary snapshot (DESIGN.md sections 1, 3, 5 and 11).

The numpy and pure-Python paths are both exercised (the latter by hiding numpy
from the module), and every store-level behaviour is run for an in-memory store
(whose scan cache is the table) and a file-backed one (SQLite is the durable
copy).
"""

import hashlib
import os
import random
import shutil
import struct
import warnings
from pathlib import Path

import pytest

import isongraph_vector as iv
from isongraph_vector import (
    EmbeddingStore,
    LoadReport,
    MockEncoder,
    SemanticGraph,
    SnapshotError,
    graph_digest,
    snapshot_path_for,
)

FIXTURES = Path(__file__).parent / "fixtures" / "snapshot"


# =============================================================================
# helpers
# =============================================================================

def make_store(mode, tmp_path, **kwargs):
    """An in-memory store or a file-backed one."""
    if mode == "memory":
        return EmbeddingStore(db_path=":memory:", **kwargs)
    return EmbeddingStore(db_path=str(tmp_path / "emb.db"), **kwargs)


MODES = ["memory", "file"]


def rand_vec(rng, dim):
    return [rng.uniform(-1, 1) for _ in range(dim)]


def brute(store, query, top_k, node_type, threshold):
    """The definition of a bounded exact search: canonical score of every live
    row, filtered, ranked by (score desc, row asc)."""
    np = pytest.importorskip("numpy")
    cache = store._ensure_cache()
    if cache.n - cache.dead == 0:
        return []
    q32, qnorm = store._query_parts(query)
    rows = np.flatnonzero(cache.alive[:cache.n])
    if node_type is not None:
        code = cache.code_of.get(node_type)
        if code is None:
            return []
        rows = rows[cache.codes[rows] == code]
    scores = store._canonical_scores(cache, q32, qnorm, rows)
    keep = scores.astype(np.float64) >= threshold
    rows, scores = rows[keep], scores[keep]
    order = np.argsort(-scores, kind="stable")
    if top_k is not None:
        order = order[:top_k]
    return [(cache.refs[i], float(s), cache.texts[i])
            for i, s in zip(rows[order].tolist(), scores[order].tolist())]


def as_tuples(results):
    return [(r.node_ref, r.score, r.text) for r in results]


def fill(store, rng, n, dim, types=("a", "b", "c"), dup_every=7, zero_every=29):
    base = rand_vec(rng, dim)
    entries = []
    for i in range(n):
        if i % dup_every == 3:
            v = list(base)                       # exact duplicates: ties
        elif i % zero_every == 5:
            v = [0.0] * dim
        elif i % 11 == 4:
            v = [x + 1e-6 * rng.uniform(-1, 1) for x in base]    # near ties
        else:
            v = rand_vec(rng, dim)
        entries.append(((types[i % len(types)], i if i % 3 else f"s{i}"), f"t{i}", v))
    store.add_batch(entries)
    return base


# =============================================================================
# 1. one copy of every vector
# =============================================================================

class TestSingleCopy:
    def test_in_memory_store_keeps_rows_only_in_the_scan_cache(self):
        pytest.importorskip("numpy")
        store = EmbeddingStore(db_path=":memory:", dimension=8)
        store.add_batch([(("p", i), f"t{i}", [float(i + 1)] * 8) for i in range(50)])
        assert store._table_mode()
        assert store._conn is None, "no SQLite connection is opened for an in-memory store"
        assert store.count() == 50
        cache = store._cache
        assert cache.matrix is not None and cache.vectors is None
        assert cache.matrix.dtype.name == "float32" and cache.matrix.flags["C_CONTIGUOUS"]
        # the private table, if something does ask for it, holds no rows
        assert store._get_conn().execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0

    def test_file_backed_store_keeps_sqlite_as_the_durable_copy(self, tmp_path):
        pytest.importorskip("numpy")
        path = str(tmp_path / "e.db")
        store = EmbeddingStore(db_path=path, dimension=4)
        store.add_batch([(("p", i), "x", [1.0, i, 0.0, 1.0]) for i in range(30)])
        store.remove(("p", 3))
        store.close()
        again = EmbeddingStore(db_path=path)
        assert again.count() == 29 and again.dimension == 4
        assert again.get(("p", 4)).vector == [1.0, 4.0, 0.0, 1.0]
        assert again.get(("p", 3)) is None
        top = again.similarity_search([1.0, 29.0, 0.0, 1.0], 1, threshold=-1)[0]
        assert top.node_ref == ("p", 29)
        again.close()

    @pytest.mark.parametrize("mode", MODES)
    def test_records_round_trip_through_every_accessor(self, mode, tmp_path):
        enc = MockEncoder(6)
        store = make_store(mode, tmp_path, encoder=enc, dimension=6)
        store.add(("person", 7), "seven", vector=[0.5] * 6)               # int id
        store.add(("person", "007"), "bond", vector=[1, 2, 3, 4, 5, 6])   # numeric-looking str id
        store.add(("doc", True), "flag", vector=[0.25] * 6)               # bool id -> "True"
        rec = store.get(("person", 7))
        assert rec.node_ref == ("person", 7) and isinstance(rec.node_ref[1], int)
        assert rec.vector == [0.5] * 6 and rec.text == "seven" and rec.model == store._model_name()
        assert store.get(("person", "007")).node_ref == ("person", "007")
        assert store.get(("person", "007")).vector == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert store.get(("doc", True)).node_ref == ("doc", "True")
        assert store.get(("nope", 1)) is None
        assert store.count() == 3
        exported = store.export_embeddings()["embeddings"]
        assert [(e["type"], e["id"], e["id_type"]) for e in exported] == [
            ("person", 7, "int"), ("person", "007", "str"), ("doc", "True", "str")]
        assert store.models() == {store._model_name(): 3}
        assert store.remove(("person", 7)) and not store.remove(("person", 7))
        assert store.count() == 2 and store.get(("person", 7)) is None
        assert store.stale() == []
        store.close()

    @pytest.mark.parametrize("mode", MODES)
    def test_a_rejected_vector_changes_nothing(self, mode, tmp_path):
        store = make_store(mode, tmp_path, dimension=3)
        store.add_batch([(("a", i), "t", [1.0, float(i), 0.0]) for i in range(10)])
        for bad in ([1.0, 2.0], [1.0, "x", 2.0], [None, 0.0, 0.0]):
            with pytest.raises(Exception):
                store.add(("a", 99), "bad", vector=bad)
        assert store.count() == 10 and store.get(("a", 99)) is None
        assert len(store.similarity_search([1, 0, 0], top_k=None, threshold=-1)) == 10
        store.close()

    def test_growth_is_amortised_and_a_bulk_write_allocates_once(self):
        np = pytest.importorskip("numpy")
        store = EmbeddingStore(db_path=":memory:", dimension=4)
        store.add(("g", 0), "x", vector=[1.0, 0.0, 0.0, 0.0])
        small = store._cache.matrix.shape[0]
        store.add_batch([(("g", i), "x", [1.0, float(i), 0.0, 0.0]) for i in range(1, 20_000)])
        cache = store._cache
        assert cache.n == 20_000 and cache.matrix.shape[0] >= 20_000 > small
        # one allocation for the batch: no more than the batch itself plus the first chunk
        assert cache.matrix.shape[0] <= 20_000 + 1024
        assert cache.matrix.base is None or cache.matrix.flags["OWNDATA"]
        assert np.array_equal(cache.matrix[5], np.array([1.0, 5.0, 0.0, 0.0], dtype=np.float32))

    @pytest.mark.parametrize("mode", MODES)
    def test_random_writes_match_a_reference_model(self, mode, tmp_path):
        pytest.importorskip("numpy")
        rng = random.Random(41)
        dim = 7
        store = make_store(mode, tmp_path, dimension=dim)
        model = {}                                # insertion-ordered like the table
        for step in range(1500):
            r = rng.random()
            if r < 0.5 or not model:
                ref = (rng.choice("abc"), rng.randrange(400) if rng.random() < .6 else f"k{rng.randrange(400)}")
                v = rand_vec(rng, dim)
                store.add(ref, f"v{step}", vector=v)
                if ref in model:
                    model[ref] = (v, f"v{step}")
                else:
                    model[ref] = (v, f"v{step}")
            elif r < 0.8:
                ref = rng.choice(list(model))
                assert store.remove(ref)
                del model[ref]
            else:
                q = rand_vec(rng, dim)
                got = store.similarity_search(q, 5, threshold=-1.0)
                assert as_tuples(got) == brute(store, q, 5, None, -1.0)
        assert store.count() == len(model)
        for ref, (v, text) in model.items():
            rec = store.get(ref)
            assert rec.text == text and rec.vector == pytest.approx(v, abs=1e-6)
        assert [(e["type"], e["id"]) for e in store.export_embeddings()["embeddings"]] \
            == list(model)
        store.close()

    @pytest.mark.parametrize("mode", MODES)
    def test_pure_python_form_is_one_copy_and_agrees_with_numpy(self, mode, tmp_path, monkeypatch):
        pytest.importorskip("numpy")
        rng = random.Random(3)
        dim = 9
        data = [((("a", "b")[i % 2], i), f"t{i}", rand_vec(rng, dim)) for i in range(120)]
        fast = make_store(mode, tmp_path, dimension=dim)
        fast.add_batch(data)
        queries = [rand_vec(rng, dim) for _ in range(6)]
        want = [[(r.node_ref, r.text) for r in fast.similarity_search(q, 8, threshold=-1)]
                for q in queries]
        fast.close()
        monkeypatch.setattr(iv, "_np", None)
        slow = EmbeddingStore(db_path=":memory:", dimension=dim)
        slow.add_batch(data)
        cache = slow._cache
        assert cache.matrix is None and cache.pos is not None
        assert all(v.typecode == "f" for v in cache.vectors)          # compact: 4 bytes a float
        for q, expected in zip(queries, want):
            got = [(r.node_ref, r.text) for r in slow.similarity_search(q, 8, threshold=-1)]
            assert got == expected
        # writes patch the list form in place (no rebuild), tombstones and compaction included
        slow.remove(("a", 2))
        slow.add(("a", 4), "again", vector=[1.0] * dim)
        for i in range(0, 120, 2):
            if i != 4:
                slow.remove(("a", i))
        assert slow._cache is cache                  # patched, never rebuilt
        assert slow.count() == 61 == len(slow.export_embeddings()["embeddings"])   # ("a", 4) + 60 "b"
        assert slow.get(("a", 4)).text == "again"
        assert slow.get(("a", 6)) is None
        top = slow.similarity_search([1.0] * dim, 1, threshold=-1)[0]
        assert top.node_ref == ("a", 4) and top.score == pytest.approx(1.0, abs=1e-6)
        slow.close()


# =============================================================================
# 2. the bounded exact search and the batch search
# =============================================================================

class TestCanonicalSearchAndBatch:
    @pytest.mark.parametrize("mode", MODES)
    def test_bounded_search_is_the_definition_on_adversarial_data(self, mode, tmp_path):
        pytest.importorskip("numpy")
        rng = random.Random(5)
        dim = 24
        store = make_store(mode, tmp_path, dimension=dim)
        base = fill(store, rng, 700, dim)
        for i in range(0, 700, 5):                                # deletes and updates
            if i % 2:
                store.remove((("a", "b", "c")[i % 3], i if i % 3 else f"s{i}"))
        queries = [rand_vec(rng, dim) for _ in range(8)] + [base, [0.0] * dim]
        # thresholds equal to scores that really occur, so the candidate set is tight
        cache = store._ensure_cache()
        q32, qn = store._query_parts(queries[0])
        occurring = sorted(set(store._canonical_scores(
            cache, q32, qn, __import__("numpy").arange(cache.n)).tolist()))
        thresholds = [-1.0, 0.0, 0.2, occurring[len(occurring) // 2], occurring[-3], 2.0]
        for q in queries:
            for top_k in (1, 3, 10, 64, 100000, None):
                for ntype in (None, "a", "zzz"):
                    for thr in thresholds:
                        got = store.similarity_search(q, top_k, ntype, thr)
                        assert as_tuples(got) == brute(store, q, top_k, ntype, thr), \
                            (top_k, ntype, thr)
        store.close()

    @pytest.mark.parametrize("mode", MODES)
    def test_canonical_scores_agree_with_float64_to_float32_precision(self, mode, tmp_path):
        np = pytest.importorskip("numpy")
        rng = random.Random(8)
        dim = 64
        store = make_store(mode, tmp_path, dimension=dim)
        fill(store, rng, 300, dim)
        q = rand_vec(rng, dim)
        for r in store.similarity_search(q, 20, threshold=-1.0):
            v = np.array(store.get(r.node_ref).vector, dtype=np.float64)
            qq = np.array(q, dtype=np.float64)
            denom = np.linalg.norm(v) * np.linalg.norm(qq)
            expected = float(v @ qq / denom) if denom else 0.0
            assert r.score == pytest.approx(expected, abs=2e-6)
        store.close()

    @pytest.mark.parametrize("mode", MODES)
    def test_batch_equals_single_bit_for_bit(self, mode, tmp_path):
        pytest.importorskip("numpy")
        rng = random.Random(6)
        dim = 16
        store = make_store(mode, tmp_path, dimension=dim, encoder=MockEncoder(dim))
        base = fill(store, rng, 900, dim)
        for i in range(0, 900, 6):
            store.remove((("a", "b", "c")[i % 3], i if i % 3 else f"s{i}"))
        queries = [rand_vec(rng, dim) for _ in range(150)] + [base, [0.0] * dim, "some text", "other"]
        for top_k in (1, 5, 10, 100000):
            for ntype in (None, "b", "zzz"):
                for thr in (-1.0, 0.0, 0.4, 0.95):
                    batch = store.similarity_search_batch(queries, top_k, ntype, thr)
                    assert len(batch) == len(queries)
                    for q, got in zip(queries, batch):
                        assert as_tuples(got) == as_tuples(
                            store.similarity_search(q, top_k, ntype, thr))
        store.close()

    def test_batch_edge_cases(self):
        pytest.importorskip("numpy")
        store = EmbeddingStore(db_path=":memory:", dimension=3)
        assert store.similarity_search_batch([]) == []
        assert store.similarity_search_batch([[1, 0, 0], [0, 1, 0]]) == [[], []]       # empty store
        store.add(("x", 1), "one", vector=[1.0, 0.0, 0.0])
        assert [[r.node_ref for r in res] for res in
                store.similarity_search_batch([[1, 0, 0]] * 3, top_k=5)] == [[("x", 1)]] * 3
        assert store.similarity_search_batch([[1, 0, 0]], top_k=0) == [[]]
        with pytest.raises(iv.DimensionMismatchError):
            store.similarity_search_batch([[1, 0, 0], [1, 0]])
        with pytest.raises(iv.NoEncoderError):
            store.similarity_search_batch(["text"])
        # one result list per query, in query order, with top_k=None meaning every match
        res = store.similarity_search_batch([[1, 0, 0], [0, 1, 0]], top_k=None, threshold=-1)
        assert [len(r) for r in res] == [1, 1]
        store.close()

    def test_batch_groups_split_large_batches_without_changing_answers(self):
        pytest.importorskip("numpy")
        rng = random.Random(1)
        dim = 8
        store = EmbeddingStore(db_path=":memory:", dimension=dim)
        fill(store, rng, 400, dim)
        store._BATCH_SCORE_BYTES = 4 * 400 * 3            # three queries per matrix product
        queries = [rand_vec(rng, dim) for _ in range(10)]
        for got, q in zip(store.similarity_search_batch(queries, 7), queries):
            assert as_tuples(got) == as_tuples(store.similarity_search(q, 7))

    def test_batch_without_canonical_scores_answers_one_by_one(self):
        pytest.importorskip("numpy")
        rng = random.Random(2)
        store = EmbeddingStore(db_path=":memory:", dimension=8)
        fill(store, rng, 200, 8)
        store.canonical_scores = False
        assert store.canonical_scores is False
        queries = [rand_vec(rng, 8) for _ in range(5)]
        for got, q in zip(store.similarity_search_batch(queries, 6), queries):
            assert as_tuples(got) == as_tuples(store.similarity_search(q, 6))

    def test_raw_blas_scores_depend_on_row_position_but_canonical_ones_do_not(self):
        """Why the bounded search rescoring exists: one vector, two slots."""
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(0)
        dim = 384
        data = rng.standard_normal((6001, dim)).astype(np.float32)
        q = rng.standard_normal(dim).astype(np.float32)
        shifted = np.concatenate([data[:1], data])            # every row moves down one slot
        a = data @ q
        b = (shifted @ q)[1:]
        raw_differs = int((a.view(np.uint32) != b.view(np.uint32)).sum())
        store = EmbeddingStore(db_path=":memory:", dimension=dim)
        store.add_batch([(("r", i), "", data[i].tolist()) for i in range(6001)])
        store.add(("pad", 0), "", vector=data[0].tolist())    # a second copy of row 0 at the end
        cache = store._ensure_cache()
        q32, qn = store._query_parts(q.tolist())
        canon = store._canonical_scores(cache, q32, qn, np.array([0, cache.n - 1]))
        assert canon[0] == canon[1]
        # raw BLAS rows that moved one slot usually changed in the last bit (a few percent on
        # OpenBLAS, none on a BLAS that is position independent): nothing to assert but the bound
        assert 0 <= raw_differs <= 6001

    @pytest.mark.parametrize("mode", MODES)
    def test_batch_with_ann_off_exact_and_loop_when_approximate(self, mode, tmp_path):
        pytest.importorskip("numpy")
        rng = random.Random(9)
        dim = 12
        store = make_store(mode, tmp_path, dimension=dim, ann="ivf", ann_min_size=0,
                           ann_nlist=8, ann_nprobe=2)
        fill(store, rng, 600, dim)
        queries = [rand_vec(rng, dim) for _ in range(6)]
        for exact in (False, True):
            for got, q in zip(store.similarity_search_batch(queries, 5, exact=exact), queries):
                assert as_tuples(got) == as_tuples(store.similarity_search(q, 5, exact=exact))
        store.close()

    def test_pure_python_batch_is_the_loop(self, monkeypatch):
        monkeypatch.setattr(iv, "_np", None)
        rng = random.Random(4)
        store = EmbeddingStore(db_path=":memory:", dimension=5)
        fill(store, rng, 80, 5)
        queries = [rand_vec(rng, 5) for _ in range(4)]
        for got, q in zip(store.similarity_search_batch(queries, 4), queries):
            assert as_tuples(got) == as_tuples(store.similarity_search(q, 4))

    def test_text_queries_in_a_batch_use_the_single_query_encoding(self):
        pytest.importorskip("numpy")
        enc = MockEncoder(16)
        store = EmbeddingStore(db_path=":memory:", encoder=enc)
        store.add_batch([(("d", i), f"document number {i}") for i in range(60)])
        texts = ["document number 3", "something else", "document number 41"]
        for got, text in zip(store.similarity_search_batch(texts, 3), texts):
            assert as_tuples(got) == as_tuples(store.similarity_search(text, 3))
        assert store.similarity_search_batch(["document number 3"], 1)[0][0].node_ref == ("d", 3)


# =============================================================================
# 3. IVF: folds at write time
# =============================================================================

def _ivf_store(n=3000, dim=16, seed=1, **kw):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((30, dim)).astype(np.float32)

    def draw(count):
        which = rng.integers(0, 30, count)
        return (centers[which] + 0.3 * rng.standard_normal((count, dim))).astype(np.float32)

    kw.setdefault("ann", "ivf")
    kw.setdefault("ann_min_size", 0)
    kw.setdefault("ann_nlist", 32)
    kw.setdefault("ann_nprobe", 6)
    store = EmbeddingStore(db_path=":memory:", dimension=dim, **kw)
    data = draw(n)
    store.add_batch([(("doc", i), "", data[i].tolist()) for i in range(n)])
    return store, draw


def _check_ivf_invariants(store):
    """Every live row is in exactly one cell list (at its own cell) or pending."""
    np = pytest.importorskip("numpy")
    cache = store._cache
    ivf = cache.ivf
    n = cache.n
    listed_rows = []
    for c in range(ivf.nlist):
        part = ivf.order[ivf.offsets[c]:ivf.offsets[c + 1]]
        mine = part[ivf.assign[part] == c]
        listed_rows.extend(mine.tolist())
    assert len(listed_rows) == len(set(listed_rows)), "a row is listed twice"
    live = set(np.flatnonzero(cache.alive[:n]).tolist())
    assert set(listed_rows) | set(ivf.pending) == live
    assert not (set(listed_rows) & set(ivf.pending))
    assert all(ivf.assign[i] == -1 for i in ivf.pending)
    assert len(ivf.pending) < 256
    assert all(ivf.listed[r] for r in listed_rows)


class TestIvfFoldAtWrite:
    def test_the_write_that_fills_a_batch_folds_it(self):
        store, draw = _ivf_store()
        store.build_ann_index()
        builds = store.ann_stats()["searches"]["builds"]
        extra = draw(600)
        for i in range(255):
            store.add(("new", i), "", vector=extra[i].tolist())
        assert store.ann_stats()["pending"] == 255            # one short: nothing folded yet
        store.add(("new", 255), "", vector=extra[255].tolist())
        assert store.ann_stats()["pending"] == 0              # the 256th write folded them
        for i in range(256, 600):
            store.add(("new", i), "", vector=extra[i].tolist())
        assert store.ann_stats()["pending"] == 600 - 512
        assert store.ann_stats()["searches"]["builds"] == builds
        _check_ivf_invariants(store)

    def test_cell_lists_stay_consistent_through_inserts_updates_deletes(self):
        pytest.importorskip("numpy")
        store, draw = _ivf_store(n=2500)
        store.build_ann_index()
        rng = random.Random(12)
        extra = draw(2000)
        nxt = 0
        live = [("doc", i) for i in range(2500)]
        for step in range(1800):
            r = rng.random()
            if r < 0.55:
                store.add(("new", nxt), "", vector=extra[nxt % 2000].tolist())
                live.append(("new", nxt))
                nxt += 1
            elif r < 0.75:
                ref = rng.choice(live)                          # update: a row that is already listed
                store.add(ref, "", vector=extra[rng.randrange(2000)].tolist())
            else:
                ref = live.pop(rng.randrange(len(live)))
                store.remove(ref)
            if step % 150 == 0 and store._cache.ivf is not None:
                _check_ivf_invariants(store)
        store.similarity_search(extra[0].tolist(), 5)
        if store._cache.ivf is not None:
            _check_ivf_invariants(store)
        # and the answers: every returned row is live, once, with its exact cosine
        for q in draw(10).tolist():
            res = store.similarity_search(q, 20, threshold=-1.0)
            refs = [r.node_ref for r in res]
            assert len(refs) == len(set(refs)) and set(refs) <= set(live)

    def test_incremental_fold_equals_a_full_regroup(self):
        pytest.importorskip("numpy")
        store, draw = _ivf_store(n=2000)
        store.build_ann_index()
        extra = draw(600)
        store.add_batch([(("new", i), "", extra[i].tolist()) for i in range(600)])
        ivf = store._cache.ivf
        ivf.fold(store._cache)
        spliced = [ivf.order[ivf.offsets[c]:ivf.offsets[c + 1]].tolist() for c in range(ivf.nlist)]
        ivf.relist(store._cache.n)
        regrouped = [ivf.order[ivf.offsets[c]:ivf.offsets[c + 1]].tolist() for c in range(ivf.nlist)]
        assert spliced == regrouped

    def test_approximate_results_carry_the_exact_searchs_scores_bit_for_bit(self):
        pytest.importorskip("numpy")
        store, draw = _ivf_store(n=3000, ann_nprobe=5)
        extra = draw(400)
        store.add_batch([(("late", i), "", extra[i].tolist()) for i in range(400)])
        for q in draw(15).tolist():
            exact = {r.node_ref: r.score
                     for r in store.similarity_search(q, 100000, threshold=-1.0, exact=True)}
            before = store.ann_stats()["searches"]["approximate"]
            found = store.similarity_search(q, 10)
            assert store.ann_stats()["searches"]["approximate"] == before + 1
            assert found and all(r.score == exact[r.node_ref] for r in found)
            ordered = [(-r.score, r.node_ref) for r in found]
            assert [r.score for r in found] == sorted((r.score for r in found), reverse=True)
            assert len(set(ordered)) == len(ordered)

    def test_probing_every_cell_still_equals_the_exact_scan(self):
        pytest.importorskip("numpy")
        store, draw = _ivf_store(n=1500, ann_nprobe=32)         # nprobe == nlist
        extra = draw(700)
        store.add_batch([(("late", i), "", extra[i].tolist()) for i in range(700)])
        for q in draw(5).tolist():
            got = as_tuples(store.similarity_search(q, 10))
            want = as_tuples(store.similarity_search(q, 10, exact=True))
            assert [g[0] for g in got] == [w[0] for w in want]


# =============================================================================
# 4. binary snapshot
# =============================================================================

def _copy_golden(tmp_path):
    for f in FIXTURES.iterdir():
        shutil.copy(f, tmp_path / f.name)
    return tmp_path / "golden.ison"


def _store_state(graph):
    """Everything a store exposes, for equality between two loads."""
    store = graph.embedding_store
    cache = store._ensure_cache()
    live = [i for i in range(cache.n) if cache.refs[i] is not None]
    return {
        "export": store.export_embeddings(),
        "models": store.models(),
        "count": store.count(),
        "dimension": store.dimension,
        "norms": None if cache.matrix is None else cache.norms[live].tobytes(),
        "matrix": None if cache.matrix is None else cache.matrix[live].tobytes(),
        "meta": dict(store.last_import_meta),
    }


def _graph_load(path, **kw):
    kw.setdefault("encoder", MockEncoder(12))
    return SemanticGraph.load_with_report(path, **kw)


class TestSnapshotGolden:
    """The fixture written by the Rust crate (DESIGN.md section 11.2)."""

    def test_golden_snapshot_is_used_and_equals_the_json_load(self, tmp_path):
        path = _copy_golden(tmp_path)
        g, report = _graph_load(path)
        assert isinstance(report, LoadReport)
        assert (report.source, report.snapshot_skipped, report.embeddings) == ("snapshot", None, 36)
        assert report.seconds >= 0 and g.load_report is report
        snap_state = _store_state(g)

        snapshot_path_for(path).unlink()
        j, jreport = _graph_load(path)
        assert (jreport.source, jreport.snapshot_skipped, jreport.embeddings) == ("json", None, 36)
        assert _store_state(j) == snap_state
        # same search results, scores bit for bit
        for q in ([0.5] * 12, [1.0] + [0.0] * 11, [0.0] * 12):
            for k in (1, 5, 100):
                a = g.embedding_store.similarity_search(q, k, threshold=-1)
                b = j.embedding_store.similarity_search(q, k, threshold=-1)
                assert as_tuples(a) == as_tuples(b)
        assert g.embedding_store.last_import_meta["graph_digest"] == graph_digest(g)

    def test_golden_snapshot_without_numpy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(iv, "_np", None)
        path = _copy_golden(tmp_path)
        g, report = _graph_load(path)
        assert report.source == "snapshot" and report.embeddings == 36
        snapshot_path_for(path).unlink()
        j, _ = _graph_load(path)
        assert g.embedding_store.export_embeddings() == j.embedding_store.export_embeddings()
        a = g.embedding_store.similarity_search([0.5] * 12, 5, threshold=-1)
        b = j.embedding_store.similarity_search([0.5] * 12, 5, threshold=-1)
        assert as_tuples(a) == as_tuples(b)

    def test_golden_header_and_layout(self):
        raw = (FIXTURES / "golden.ison.embeddings.snap").read_bytes()
        assert raw[:8] == b"ISONVSNP"
        version, flags, dim, block_rows, count = struct.unpack_from("<IIIIQ", raw, 8)
        assert (version, flags, dim, block_rows, count) == (1, 1, 12, 1024, 36)
        assert hashlib.sha256(raw[:-32]).digest() == raw[-32:]
        info = iv._read_snapshot(FIXTURES / "golden.ison.embeddings.snap").info
        assert info.count == 36 and info.dimension == 12 and info.sidecar_len is not None
        side = FIXTURES / "golden.ison.embeddings.json"
        assert iv.sidecar_identity(side) == (info.sidecar_len, info.sidecar_fingerprint)


class TestSnapshotRejections:
    """A snapshot that is not provably the sidecar's twin is ignored, never half-used."""

    def _load(self, path, **kw):
        g, report = _graph_load(path, **kw)
        assert report.source == "json", report
        assert report.snapshot_skipped, "a snapshot that exists and was refused says why"
        assert report.embeddings == 36
        reference = _store_state(g)
        snapshot_path_for(path).unlink()
        j, _ = _graph_load(path, **kw)
        assert _store_state(j) == reference
        return report.snapshot_skipped

    def test_corrupt_body_byte(self, tmp_path):
        path = _copy_golden(tmp_path)
        snap = snapshot_path_for(path)
        raw = bytearray(snap.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        snap.write_bytes(bytes(raw))
        assert "checksum" in self._load(path)

    def test_every_truncation_is_refused(self, tmp_path):
        path = _copy_golden(tmp_path)
        snap = snapshot_path_for(path)
        good = snap.read_bytes()
        for cut in (0, 5, 8, 20, 33, 100, 200, len(good) // 2, len(good) - 33, len(good) - 32,
                    len(good) - 1):
            snap.write_bytes(good[:cut])
            g, report = _graph_load(path)
            assert report.source == "json" and report.snapshot_skipped, cut
            assert g.embedding_store.count() == 36

    def test_trailing_garbage_bad_magic_version_and_flags(self, tmp_path):
        path = _copy_golden(tmp_path)
        snap = snapshot_path_for(path)
        good = snap.read_bytes()

        def with_header(offset, value, fmt="<I"):
            raw = bytearray(good[:-32])
            struct.pack_into(fmt, raw, offset, value)
            return bytes(raw) + hashlib.sha256(bytes(raw)).digest()      # valid checksum, bad field

        variants = {
            "extra byte": good + b"\0",
            "bad magic": b"XSONVSNP" + good[8:],
            "version 2": with_header(8, 2),
            "unknown flag": with_header(12, 3),
            "zero block": with_header(20, 0),
            "huge block": with_header(20, (1 << 20) + 1),
            "count too big": with_header(24, 1 << 40, "<Q"),
            "count beyond file": with_header(24, 10_000, "<Q"),
            "dim zero": with_header(16, 0),
        }
        for name, data in variants.items():
            snap.write_bytes(data)
            g, report = _graph_load(path)
            assert report.source == "json" and report.snapshot_skipped, name
            assert g.embedding_store.count() == 36, name

    def test_stale_sidecar(self, tmp_path):
        path = _copy_golden(tmp_path)
        side = Path(str(path) + ".embeddings.json")
        text = side.read_text(encoding="utf-8")
        side.write_text(text + "\n", encoding="utf-8")            # the length changes
        assert "sidecar" in self._load(path)

    def test_stale_sidecar_same_length_in_a_sampled_region(self, tmp_path):
        path = _copy_golden(tmp_path)
        side = Path(str(path) + ".embeddings.json")
        data = bytearray(side.read_bytes())
        at = next(i for i in range(200, len(data)) if chr(data[i]).isdigit())   # inside the first 64 KiB
        data[at] = ord("1") if data[at] != ord("1") else ord("2")              # still valid JSON
        side.write_bytes(bytes(data))
        g, report = _graph_load(path, on_mismatch="warn")
        assert report.source == "json" and "sidecar" in report.snapshot_skipped

    def test_missing_sidecar_means_no_snapshot_use(self, tmp_path):
        path = _copy_golden(tmp_path)
        Path(str(path) + ".embeddings.json").unlink()
        g, report = _graph_load(path)
        assert report.source == "none" and g.embedding_store.count() == 0

    def test_graph_digest_mismatch(self, tmp_path):
        path = _copy_golden(tmp_path)
        path.write_text(path.read_text(encoding="utf-8").replace("Alice", "Alicia"),
                        encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, report = _graph_load(path, on_mismatch="warn")
        assert report.source == "json" and "different graph" in report.snapshot_skipped
        assert g.embedding_store.count() == 36

    def test_snapshot_naming_a_different_digest_is_refused(self, tmp_path):
        path = _copy_golden(tmp_path)
        snap = snapshot_path_for(path)
        raw = snap.read_bytes()
        digest = graph_digest(SemanticGraph.load(path, encoder=MockEncoder(12))).encode()
        assert digest in raw
        body = raw[:-32].replace(digest, digest[:-1] + (b"0" if digest[-1:] != b"0" else b"1"))
        snap.write_bytes(body + hashlib.sha256(body).digest())     # internally valid
        g, report = _graph_load(path)
        assert report.source == "json" and "different graph" in report.snapshot_skipped

    def test_duplicate_key_and_unlisted_model_are_refused(self, tmp_path):
        store = EmbeddingStore(db_path=":memory:", dimension=2)
        store.add_batch([(("t", "a"), "x", [1.0, 2.0]), (("t", "b"), "y", [3.0, 4.0])])
        snap = tmp_path / "s.snap"
        store.save_snapshot(snap)
        good = snap.read_bytes()
        body = good[:-32]
        # the second row's id "b" -> "a": a repeated key, with a valid checksum
        dup = body.replace(struct.pack("<I", 1) + b"b" + struct.pack("<I", 1) + b"y",
                           struct.pack("<I", 1) + b"a" + struct.pack("<I", 1) + b"y")
        assert dup != body
        snap.write_bytes(dup + hashlib.sha256(dup).digest())
        fresh = EmbeddingStore(db_path=":memory:")
        with pytest.raises(SnapshotError, match="twice"):
            fresh.load_snapshot(snap)
        assert fresh.count() == 0 and fresh.dimension is None
        # a model index past the model list
        idx = body.rindex(struct.pack("<I", 0) + struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0))
        bad = body[:idx] + struct.pack("<I", 5) + body[idx + 4:]
        snap.write_bytes(bad + hashlib.sha256(bad).digest())
        with pytest.raises(SnapshotError, match="model"):
            fresh.load_snapshot(snap)
        assert fresh.count() == 0

    def test_a_refused_snapshot_leaves_a_populated_store_untouched(self, tmp_path):
        src = EmbeddingStore(db_path=":memory:", dimension=2)
        src.add_batch([(("t", i), "x", [1.0, float(i)]) for i in range(20)])
        snap = tmp_path / "s.snap"
        src.save_snapshot(snap)
        raw = bytearray(snap.read_bytes())
        raw[-40] ^= 0x01                                    # inside the last row's vector
        snap.write_bytes(bytes(raw))
        target = EmbeddingStore(db_path=":memory:", dimension=2)
        target.add(("keep", 1), "kept", vector=[9.0, 9.0])
        before = target.export_embeddings()
        with pytest.raises(SnapshotError, match="checksum"):
            target.load_snapshot(snap)
        assert target.export_embeddings() == before
        with pytest.raises(SnapshotError):
            target.load_snapshot(tmp_path / "missing.snap")
        # a snapshot of another width is refused before anything is written
        wide = EmbeddingStore(db_path=":memory:", dimension=5)
        wide.add(("w", 1), "", vector=[1.0] * 5)
        wide.save_snapshot(snap)
        with pytest.raises(SnapshotError, match="dimensional"):
            target.load_snapshot(snap)
        assert target.export_embeddings() == before


class TestSnapshotRoundTrip:
    def _graph(self, with_ints=True):
        enc = MockEncoder(10)
        g = SemanticGraph(name="rt", embedding_db=":memory:", encoder=enc, auto_embed=False)
        for i in range(120):
            g.add_node("person", i, name=f"p{i}")
            g.embed_node(("person", i), f"person number {i} with some text")
        for i in range(0, 120, 3):
            g.add_node("doc", f"d{i}", name=f"d{i}")
            g.embed_node(("doc", f"d{i}"), f"doc \"quoted\" \n multi-line é中😀 {i}")
        g.add_node("acct", "007", name="bond")
        g.embed_node(("acct", "007"), "")
        g.add_edge("KNOWS", ("person", 1), ("person", 2))
        g.embedding_store.remove(("person", 5))
        g.embedding_store.add(("person", 9), "updated", vector=[0.25] * 10)
        return g

    @pytest.mark.parametrize("mode", MODES)
    def test_save_load_round_trip(self, mode, tmp_path):
        g = self._graph()
        path = tmp_path / "rt.ison"
        g.save(path, snapshot=True)
        assert snapshot_path_for(path).is_file()
        loaded, report = SemanticGraph.load_with_report(
            path, encoder=MockEncoder(10),
            embedding_db=":memory:" if mode == "memory" else str(tmp_path / "load.db"))
        assert report.source == "snapshot" and report.snapshot_skipped is None
        assert report.embeddings == g.embedding_store.count()
        # int ids are ints again, str ids that look numeric stay strings
        assert loaded.embedding_store.get(("person", 3)).node_ref == ("person", 3)
        assert isinstance(loaded.embedding_store.get(("person", 3)).node_ref[1], int)
        assert loaded.embedding_store.get(("acct", "007")).node_ref == ("acct", "007")
        assert loaded.embedding_store.get(("person", 5)) is None
        # equal to a JSON load of the same files
        snapshot_path_for(path).unlink()
        j = SemanticGraph.load(path, encoder=MockEncoder(10))
        assert loaded.embedding_store.export_embeddings() == j.embedding_store.export_embeddings()
        assert loaded.embedding_store.export_embeddings() == g.embedding_store.export_embeddings()
        q = [0.3] * 10
        assert as_tuples(loaded.embedding_store.similarity_search(q, 12, threshold=-1)) \
            == as_tuples(j.embedding_store.similarity_search(q, 12, threshold=-1))
        loaded.close()

    def test_json_sidecar_and_graph_bytes_do_not_depend_on_the_snapshot(self, tmp_path):
        g = self._graph()
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        g.save(a / "g.ison")
        g.save(b / "g.ison", snapshot=True)
        for name in ("g.ison", "g.ison.embeddings.json"):
            assert (a / name).read_bytes() == (b / name).read_bytes()
        assert not (a / "g.ison.embeddings.snap").exists()
        assert (b / "g.ison.embeddings.snap").is_file()

    def test_save_without_snapshot_deletes_the_old_one(self, tmp_path):
        g = self._graph()
        path = tmp_path / "g.ison"
        g.save(path, snapshot=True)
        assert snapshot_path_for(path).is_file()
        g.embedding_store.add(("person", 77), "changed", vector=[0.5] * 10)
        g.save(path)
        assert not snapshot_path_for(path).exists()
        loaded, report = SemanticGraph.load_with_report(path, encoder=MockEncoder(10))
        assert report.source == "json" and report.snapshot_skipped is None
        assert loaded.embedding_store.get(("person", 77)).text == "changed"

    def test_a_stale_snapshot_is_refused_after_a_graph_only_save(self, tmp_path):
        g = self._graph()
        path = tmp_path / "g.ison"
        g.save(path, snapshot=True)
        g.add_node("person", 5000, name="late")
        g.save(path, embeddings=False)               # sidecar and snapshot stay; the graph moved on
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded, report = SemanticGraph.load_with_report(
                path, encoder=MockEncoder(10), on_mismatch="warn")
        assert report.source == "json" and "different graph" in report.snapshot_skipped

    def test_failed_snapshot_write_keeps_the_pair_and_removes_the_stale_snapshot(
            self, tmp_path, monkeypatch):
        g = self._graph()
        path = tmp_path / "g.ison"
        g.save(path, snapshot=True)
        old = snapshot_path_for(path).read_bytes()
        g.embedding_store.add(("person", 77), "changed", vector=[0.5] * 10)

        def boom(src, dst):
            if str(dst).endswith(".snap"):
                raise OSError("disk full")
            return real(src, dst)

        real = os.replace
        monkeypatch.setattr(iv.os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            g.save(path, snapshot=True)
        monkeypatch.undo()
        assert not snapshot_path_for(path).exists()
        assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
        loaded = SemanticGraph.load(path, encoder=MockEncoder(10))
        assert loaded.embedding_store.get(("person", 77)).text == "changed"
        assert old != b""

    def test_ids_a_snapshot_cannot_describe_are_refused_at_save(self, tmp_path):
        g = SemanticGraph(name="ids", embedding_db=":memory:", encoder=MockEncoder(4),
                          auto_embed=False)
        g.add_node("p", 1, name="one")
        g.embedding_store.add(("p", 1), "one", vector=[1.0, 0.0, 0.0, 0.0])
        g.embedding_store.add(("ghost", 7), "no node", vector=[0.0, 1.0, 0.0, 0.0])   # int id, no node
        path = tmp_path / "g.ison"
        with pytest.raises(SnapshotError, match="integer"):
            g.save(path, snapshot=True)
        assert path.is_file() and Path(str(path) + ".embeddings.json").is_file()
        assert not snapshot_path_for(path).exists()
        loaded = SemanticGraph.load(path, encoder=MockEncoder(4))
        assert loaded.embedding_store.get(("ghost", 7)).node_ref == ("ghost", 7)

    def test_snapshot_without_numpy_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(iv, "_np", None)
        g = self._graph()
        path = tmp_path / "g.ison"
        g.save(path, snapshot=True)
        loaded, report = SemanticGraph.load_with_report(path, encoder=MockEncoder(10))
        assert report.source == "snapshot"
        assert loaded.embedding_store.export_embeddings() == g.embedding_store.export_embeddings()

    def test_numpy_reads_what_pure_python_wrote_and_back(self, tmp_path, monkeypatch):
        g = self._graph()
        a, b = tmp_path / "a.ison", tmp_path / "b.ison"
        g.save(a, snapshot=True)
        with monkeypatch.context() as m:
            m.setattr(iv, "_np", None)
            slow = self._graph()
            slow.save(b, snapshot=True)
        assert snapshot_path_for(a).read_bytes() == snapshot_path_for(b).read_bytes()

    def test_empty_store_snapshot(self, tmp_path):
        g = SemanticGraph(name="e", embedding_db=":memory:", encoder=MockEncoder(4), auto_embed=False)
        g.add_node("p", 1, name="x")
        path = tmp_path / "e.ison"
        g.save(path, snapshot=True)
        loaded, report = SemanticGraph.load_with_report(path, encoder=MockEncoder(4))
        assert report.source == "snapshot" and report.embeddings == 0

    def test_minimal_example_from_the_spec_is_byte_exact(self, tmp_path):
        store = EmbeddingStore(db_path=":memory:", dimension=2)
        store.add(("t", "a"), "x", vector=[1.0, -2.0])
        path = tmp_path / "min.snap"
        info = store.save_snapshot(path)
        raw = path.read_bytes()
        assert info.bytes == len(raw) == 123
        expected = bytes.fromhex(
            "49534f4e56534e50" "01000000" "00000000"
            "02000000" "00040000" "0100000000000000"
            "ffffffff" "ffffffff" "ffffffff" "ffffffff"
            "01000000" "08000000") + b"provided" + bytes.fromhex(
            "01000000") + b"t" + bytes.fromhex("01000000") + b"a" + bytes.fromhex(
            "01000000") + b"x" + bytes.fromhex("00000000" "0000803f" "000000c0")
        assert raw[:-32] == expected and len(expected) == 91
        assert raw[-32:] == hashlib.sha256(expected).digest()

    def test_large_snapshot_spans_blocks(self, tmp_path):
        pytest.importorskip("numpy")
        rng = random.Random(21)
        store = EmbeddingStore(db_path=":memory:", dimension=6)
        store.add_batch([(("t", i), f"text {i}", rand_vec(rng, 6)) for i in range(2500)])
        path = tmp_path / "big.snap"
        store.save_snapshot(path)
        other = EmbeddingStore(db_path=":memory:")
        other.load_snapshot(path)
        assert other.count() == 2500
        assert other.export_embeddings()["embeddings"] == [
            dict(e, id=str(e["id"]), id_type="str") for e in store.export_embeddings()["embeddings"]]
        q = rand_vec(rng, 6)
        a = [(r.node_ref[1], r.score) for r in store.similarity_search(q, 10)]
        b = [(int(r.node_ref[1]), r.score) for r in other.similarity_search(q, 10)]
        assert a == b
        # load into a populated store merges, overwriting equal keys
        other.load_snapshot(path)
        assert other.count() == 2500
        other.load_snapshot(path, replace=True)
        assert other.count() == 2500

    def test_writes_after_a_snapshot_load_work(self, tmp_path):
        g = self._graph()
        path = tmp_path / "g.ison"
        g.save(path, snapshot=True)
        loaded = SemanticGraph.load(path, encoder=MockEncoder(10))
        store = loaded.embedding_store
        n = store.count()
        for i in range(3000):
            store.add(("extra", i), "", vector=[float(i % 7), 1.0] + [0.1] * 8)
        assert store.count() == n + 3000
        store.remove(("person", 3))
        assert store.get(("person", 3)) is None and store.count() == n + 2999
        assert store.similarity_search([6.0, 1.0] + [0.1] * 8, 1)[0].node_ref[0] == "extra"
