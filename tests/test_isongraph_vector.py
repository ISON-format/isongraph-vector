#!/usr/bin/env python3
"""
Tests for isongraph-vector

Uses MockEncoder by default to avoid sentence-transformers dependency in CI.
Set USE_REAL_ENCODER=1 to test with actual SentenceTransformer.
"""

import importlib.util
import os
import random
import threading
import pytest
import tempfile
from pathlib import Path

# Use mock encoder by default
USE_REAL_ENCODER = os.environ.get('USE_REAL_ENCODER', '0') == '1'

from isongraph_vector import (
    SemanticGraph,
    EmbeddingStore,
    MockEncoder,
    SentenceTransformerEncoder,
    SimilarityResult,
    SemanticSearchResult,
    DimensionMismatchError,
    NoEncoderError,
    EMBEDDING_FORMAT,
)
from ison_graph import Direction


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def encoder():
    """Get encoder based on environment."""
    if USE_REAL_ENCODER:
        return SentenceTransformerEncoder()
    return MockEncoder(dimension=384)


@pytest.fixture
def embedding_store(encoder):
    """Create in-memory embedding store."""
    return EmbeddingStore(db_path=":memory:", encoder=encoder)


@pytest.fixture
def semantic_graph(encoder):
    """Create semantic graph with mock encoder."""
    graph = SemanticGraph(
        name="test",
        embedding_db=":memory:",
        encoder=encoder,
        auto_embed=False
    )
    return graph


@pytest.fixture
def populated_graph(encoder):
    """Create graph with test data."""
    graph = SemanticGraph(
        name="social",
        embedding_db=":memory:",
        encoder=encoder,
        auto_embed=False
    )

    # Add people
    graph.add_node('person', 1, name='Alice', role='Software Engineer')
    graph.add_node('person', 2, name='Bob', role='Data Scientist')
    graph.add_node('person', 3, name='Charlie', role='Product Manager')
    graph.add_node('person', 4, name='Diana', role='UX Designer')

    # Add company
    graph.add_node('company', 100, name='TechCorp', industry='Technology')

    # Add relationships
    graph.add_edge('KNOWS', ('person', 1), ('person', 2))
    graph.add_edge('KNOWS', ('person', 2), ('person', 3))
    graph.add_edge('KNOWS', ('person', 3), ('person', 4))
    graph.add_edge('WORKS_AT', ('person', 1), ('company', 100))
    graph.add_edge('WORKS_AT', ('person', 2), ('company', 100))

    # Add embeddings
    graph.embed_node(('person', 1), 'Alice is a software engineer who writes code')
    graph.embed_node(('person', 2), 'Bob is a data scientist who analyzes data')
    graph.embed_node(('person', 3), 'Charlie is a product manager who plans features')
    graph.embed_node(('person', 4), 'Diana is a UX designer who creates interfaces')
    graph.embed_node(('company', 100), 'TechCorp is a technology company')

    return graph


# =============================================================================
# EmbeddingStore Tests
# =============================================================================

class TestEmbeddingStore:
    """Tests for EmbeddingStore."""

    def test_add_and_get(self, embedding_store):
        """Test adding and retrieving embeddings."""
        embedding_store.add(('person', 1), 'Alice is a software engineer')

        record = embedding_store.get(('person', 1))
        assert record is not None
        assert record.node_ref == ('person', 1)
        assert isinstance(record.node_ref[1], int)
        assert record.text == 'Alice is a software engineer'
        assert len(record.vector) == 384

    def test_string_node_id_round_trips_exactly(self, embedding_store):
        """node_id used to round-trip through an unsound int(node_id)
        guess in similarity_search() (get() didn't even try - it always
        returned a string, even for int ids added above). A numeric-
        looking string id like "007" - a real shape for account numbers,
        zip codes, etc. - silently became the integer 7, which then made
        get_node_by_ref() fail to find the actual node in SemanticGraph
        usage (see test_semantic_multi_hop_preserves_string_node_id
        below). Fixed via an explicit node_id_type column instead of
        guessing; both get() and similarity_search() must now agree."""
        embedding_store.add(('account', '007'), 'special account zero-zero-seven')

        record = embedding_store.get(('account', '007'))
        assert record.node_ref == ('account', '007')
        assert isinstance(record.node_ref[1], str)

        results = embedding_store.similarity_search('special account zero-zero-seven', top_k=1)
        assert results[0].node_ref == ('account', '007')
        assert isinstance(results[0].node_ref[1], str)

    def test_add_with_vector(self, embedding_store):
        """Test adding pre-computed vector."""
        vector = [0.1] * 384
        embedding_store.add(('person', 1), 'test', vector=vector)

        record = embedding_store.get(('person', 1))
        assert record is not None
        # Use approximate comparison due to float32 storage precision
        assert record.vector[:5] == pytest.approx([0.1, 0.1, 0.1, 0.1, 0.1], rel=1e-5)

    def test_remove(self, embedding_store):
        """Test removing embeddings."""
        embedding_store.add(('person', 1), 'test')
        assert embedding_store.get(('person', 1)) is not None

        embedding_store.remove(('person', 1))
        assert embedding_store.get(('person', 1)) is None

    def test_similarity_search(self, embedding_store):
        """Test similarity search."""
        embedding_store.add(('person', 1), 'software engineer')
        embedding_store.add(('person', 2), 'data scientist')
        embedding_store.add(('person', 3), 'product manager')

        results = embedding_store.similarity_search('engineer', top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SimilarityResult) for r in results)

    def test_similarity_search_with_type_filter(self, embedding_store):
        """Test filtering by node type."""
        embedding_store.add(('person', 1), 'software engineer')
        embedding_store.add(('company', 100), 'engineering company')

        results = embedding_store.similarity_search('engineer', node_type='person')
        assert all(r.node_ref[0] == 'person' for r in results)

    def test_count(self, embedding_store):
        """Test counting embeddings."""
        assert embedding_store.count() == 0

        embedding_store.add(('person', 1), 'test1')
        embedding_store.add(('person', 2), 'test2')

        assert embedding_store.count() == 2

    def test_clear(self, embedding_store):
        """Test clearing all embeddings."""
        embedding_store.add(('person', 1), 'test1')
        embedding_store.add(('person', 2), 'test2')

        embedding_store.clear()
        assert embedding_store.count() == 0

    def test_persistence(self, encoder):
        """Test persisting to file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            # Create and populate
            store1 = EmbeddingStore(db_path=db_path, encoder=encoder)
            store1.add(('person', 1), 'Alice')
            store1.close()

            # Reload and verify
            store2 = EmbeddingStore(db_path=db_path, encoder=encoder)
            record = store2.get(('person', 1))
            assert record is not None
            assert record.text == 'Alice'
            store2.close()
        finally:
            Path(db_path).unlink(missing_ok=True)


# =============================================================================
# SemanticGraph Tests
# =============================================================================

class TestSemanticGraph:
    """Tests for SemanticGraph."""

    def test_create_empty(self, encoder):
        """Test creating empty graph."""
        graph = SemanticGraph(encoder=encoder)
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_add_node_with_embed(self, semantic_graph):
        """Test adding node with embedding."""
        semantic_graph.add_node(
            'person', 1,
            name='Alice',
            _embed_text='Alice is a software engineer'
        )

        assert semantic_graph.has_node('person', 1)
        assert semantic_graph.get_embedding(('person', 1)) is not None

    def test_auto_embed(self, encoder):
        """Test auto-embedding from fields."""
        graph = SemanticGraph(
            encoder=encoder,
            auto_embed=True,
            embed_fields=['name', 'description']
        )

        graph.add_node('person', 1, name='Alice', description='Engineer')

        record = graph.get_embedding(('person', 1))
        assert record is not None
        assert 'Alice' in record.text
        assert 'Engineer' in record.text

    def test_remove_node_removes_embedding(self, semantic_graph):
        """Test that removing node removes its embedding."""
        semantic_graph.add_node('person', 1, _embed_text='test')
        assert semantic_graph.get_embedding(('person', 1)) is not None

        semantic_graph.remove_node('person', 1)
        assert semantic_graph.get_embedding(('person', 1)) is None

    def test_similarity_search(self, populated_graph):
        """Test similarity search on graph."""
        results = populated_graph.similarity_search('software code', top_k=3)

        assert len(results) > 0
        # First result should be most similar to 'software code'
        assert isinstance(results[0], SimilarityResult)

    def test_semantic_multi_hop(self, populated_graph):
        """Test semantic multi-hop search."""
        results = populated_graph.semantic_multi_hop(
            query='software engineer',
            rel_type='KNOWS',
            max_hops=2,
            top_k_seeds=3,
            top_k_results=10,
            # Cosine similarity is signed and MockEncoder vectors are
            # near-orthogonal, so mock scores straddle zero - a threshold of
            # 0.0 can legitimately reject every seed. Match the Rust port and
            # accept anything.
            threshold=-1.0
        )

        assert len(results) > 0
        assert all(isinstance(r, SemanticSearchResult) for r in results)
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_semantic_multi_hop_with_decay(self, encoder):
        """Score decays by decay ** hop_count along a chain.

        Hand-built vectors rather than encoder output: the exact scores have
        to be known for the assertion to mean anything. MockEncoder vectors
        are near-orthogonal noise, so an earlier version of this test only
        compared two arbitrary near-zero numbers and passed by luck.
        """
        graph = SemanticGraph(name='chain', embedding_db=':memory:',
                              encoder=encoder, auto_embed=False)
        for i in range(1, 5):
            graph.add_node('person', i, name=f'P{i}')
        for i in range(1, 4):
            graph.add_edge('KNOWS', ('person', i), ('person', i + 1))

        # person:1 matches the query exactly; the rest are orthogonal to it.
        graph.embedding_store.add(('person', 1), 'seed', vector=[1.0, 0.0, 0.0, 0.0])
        for i in range(2, 5):
            graph.embedding_store.add(('person', i), f'p{i}', vector=[0.0, 1.0, 0.0, 0.0])

        results = graph.semantic_multi_hop(
            query=[1.0, 0.0, 0.0, 0.0],
            rel_type='KNOWS',
            max_hops=3,
            decay=0.5,
            threshold=0.5,          # only person:1 seeds the traversal
        )

        by_ref = {r.node_ref: r for r in results}
        assert by_ref[('person', 1)].score == pytest.approx(1.0)
        assert by_ref[('person', 2)].score == pytest.approx(0.5)
        assert by_ref[('person', 3)].score == pytest.approx(0.25)
        assert by_ref[('person', 4)].score == pytest.approx(0.125)
        assert [by_ref[('person', i)].hop_count for i in range(1, 5)] == [0, 1, 2, 3]

        # Scores strictly decrease with distance.
        scores = [by_ref[('person', i)].score for i in range(1, 5)]
        assert scores == sorted(scores, reverse=True)

    def test_semantic_multi_hop_finds_node_with_string_id(self, encoder):
        """similarity_search() used to guess node id types via
        int(node_id), silently turning a numeric-looking string id like
        "007" into the integer 7. Since semantic_multi_hop() calls
        get_node_by_ref() with whatever ref similarity_search() hands it,
        this made it silently fail to find the very node it just
        matched: get_node_by_ref(('account', 7)) doesn't find a node
        actually stored as ('account', '007'), so `.node` came back None
        even though the search "succeeded". Verified directly against
        the old code before fixing it."""
        graph = SemanticGraph(name="strid", embedding_db=":memory:", encoder=encoder, auto_embed=False)
        graph.add_node('account', '007', name='Special Account', _embed_text='special account zero zero seven')

        results = graph.semantic_multi_hop(
            query='special account zero zero seven',
            max_hops=1,
            threshold=0.0,
        )
        assert len(results) == 1
        assert results[0].node_ref == ('account', '007')
        assert results[0].node is not None
        assert results[0].node.properties['name'] == 'Special Account'

    def test_semantic_path(self, populated_graph):
        """Test finding semantically-rooted path."""
        result = populated_graph.semantic_path(
            query='software engineer',
            target_ref=('person', 3),
            rel_type='KNOWS',
            max_hops=3,
            threshold=-1.0  # see test_semantic_multi_hop
        )

        assert result is not None
        assert result.node_ref == ('person', 3)
        assert len(result.path) > 0

    def test_embed_all_nodes(self, semantic_graph):
        """Test batch embedding."""
        semantic_graph.add_node('person', 1, name='Alice')
        semantic_graph.add_node('person', 2, name='Bob')
        semantic_graph.add_node('person', 3, name='Charlie')

        count = semantic_graph.embed_all_nodes()
        assert count == 3
        assert semantic_graph.embedding_store.count() == 3

    def test_graph_traversal_still_works(self, populated_graph):
        """Test that base graph operations still work."""
        # Test neighbors
        neighbors = populated_graph.neighbors(('person', 1), 'KNOWS')
        assert ('person', 2) in neighbors

        # Test multi-hop
        two_hops = populated_graph.multi_hop(('person', 1), 'KNOWS', hops=2)
        assert ('person', 3) in two_hops

        # Test path
        path = populated_graph.shortest_path(('person', 1), ('person', 3))
        assert path is not None
        assert path.length == 2

    def test_save_and_load(self, populated_graph):
        """Test saving and loading graph."""
        with tempfile.NamedTemporaryFile(suffix='.isong', delete=False) as f:
            graph_path = f.name

        try:
            # Save
            populated_graph.save(graph_path)

            # Load (embeddings stored separately)
            loaded = SemanticGraph.load(graph_path)

            assert loaded.node_count() == populated_graph.node_count()
            assert loaded.edge_count() == populated_graph.edge_count()
        finally:
            Path(graph_path).unlink(missing_ok=True)


# =============================================================================
# MockEncoder Tests
# =============================================================================

class TestMockEncoder:
    """Tests for MockEncoder."""

    def test_deterministic(self):
        """Test that encoding is deterministic."""
        encoder = MockEncoder()
        v1 = encoder.encode("test")
        v2 = encoder.encode("test")
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        """Test that different texts produce different vectors."""
        encoder = MockEncoder()
        v1 = encoder.encode("hello")
        v2 = encoder.encode("world")
        assert v1 != v2

    def test_dimension(self):
        """Test custom dimension."""
        encoder = MockEncoder(dimension=512)
        v = encoder.encode("test")
        assert len(v) == 512

    def test_batch_encode(self):
        """Test batch encoding."""
        encoder = MockEncoder()
        texts = ["one", "two", "three"]
        vectors = encoder.encode_batch(texts)
        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self, encoder):
        """Test complete workflow."""
        # Create graph
        graph = SemanticGraph(
            name="test_workflow",
            encoder=encoder,
            auto_embed=True,
            embed_fields=['description']
        )

        # Add nodes
        graph.add_node('article', 1, title='Python Guide', description='A guide to Python programming')
        graph.add_node('article', 2, title='ML Intro', description='Introduction to machine learning')
        graph.add_node('article', 3, title='Web Dev', description='Web development with JavaScript')
        graph.add_node('topic', 'python', name='Python')
        graph.add_node('topic', 'ml', name='Machine Learning')

        # Add relationships
        graph.add_edge('ABOUT', ('article', 1), ('topic', 'python'))
        graph.add_edge('ABOUT', ('article', 2), ('topic', 'ml'))
        graph.add_edge('RELATED', ('article', 1), ('article', 2))

        # Embed topics manually
        graph.embed_node(('topic', 'python'), 'Python programming language')
        graph.embed_node(('topic', 'ml'), 'Machine learning AI')

        # Search
        results = graph.similarity_search('programming language', top_k=3)
        assert len(results) > 0

        # Semantic multi-hop
        results = graph.semantic_multi_hop(
            query='Python development',
            max_hops=2,
            threshold=0.0  # Low threshold for mock encoder
        )
        assert len(results) > 0

        # Verify graph structure intact
        assert graph.node_count() == 5
        assert graph.edge_count() == 3


# =============================================================================
# Regression tests for fixed defects
# =============================================================================

class TestSeedScoring:
    """semantic_multi_hop must not downgrade a node that matches directly."""

    @staticmethod
    def _two_node_graph(encoder):
        graph = SemanticGraph(name='seeds', embedding_db=':memory:',
                              encoder=encoder, auto_embed=False)
        graph.add_node('person', 1, name='A')
        graph.add_node('person', 2, name='B')
        graph.add_edge('KNOWS', ('person', 1), ('person', 2))
        # person:1 matches the query perfectly, person:2 slightly less well.
        graph.embedding_store.add(('person', 1), 'a', vector=[1.0, 0.0, 0.0, 0.0])
        graph.embedding_store.add(('person', 2), 'b', vector=[0.9, 0.19 ** 0.5, 0.0, 0.0])
        return graph

    def test_seed_keeps_its_own_score(self, encoder):
        """A seed reached from a better seed used to keep the decayed score.

        Seeds were inserted only when the ref was absent from results, so an
        entry written by an earlier traversal won even when it scored lower.
        person:2 is itself a 0.9 match at hop 0, but came back at
        seed(1.0) * decay(0.8) = 0.8, at hop 1, down a path through person:1
        that it never needed.
        """
        graph = self._two_node_graph(encoder)
        query = [1.0, 0.0, 0.0, 0.0]

        direct = {r.node_ref: r.score for r in graph.similarity_search(query, top_k=5)}
        assert direct[('person', 2)] == pytest.approx(0.9, abs=1e-6)

        results = {r.node_ref: r for r in graph.semantic_multi_hop(
            query, rel_type='KNOWS', max_hops=2, decay=0.8, threshold=0.3
        )}

        assert results[('person', 2)].score == pytest.approx(0.9, abs=1e-6)
        assert results[('person', 2)].hop_count == 0
        assert results[('person', 2)].path == [('person', 2)]

    def test_traversal_still_reaches_non_seeds(self, encoder):
        """The seed fix must not stop traversal from scoring neighbours."""
        graph = self._two_node_graph(encoder)
        graph.add_node('person', 3, name='C')
        graph.add_edge('KNOWS', ('person', 2), ('person', 3))
        graph.embedding_store.add(('person', 3), 'c', vector=[0.0, 0.0, 1.0, 0.0])

        results = {r.node_ref: r for r in graph.semantic_multi_hop(
            [1.0, 0.0, 0.0, 0.0], rel_type='KNOWS', max_hops=2,
            decay=0.8, threshold=0.3
        )}
        # person:3 is orthogonal to the query - only reachable by traversal.
        assert results[('person', 3)].hop_count in (1, 2)
        assert results[('person', 3)].score > 0

    def test_blend_lifts_a_relevant_distant_node(self, encoder):
        """blend mixes the relevance of a node itself into its hop score."""
        graph = SemanticGraph(name='blend', embedding_db=':memory:',
                              encoder=encoder, auto_embed=False)
        for i in (1, 2, 3):
            graph.add_node('person', i, name=f'P{i}')
        graph.add_edge('KNOWS', ('person', 1), ('person', 2))
        graph.add_edge('KNOWS', ('person', 1), ('person', 3))
        query = [1.0, 0.0, 0.0, 0.0]
        graph.embedding_store.add(('person', 1), 'seed', vector=[1.0, 0.0, 0.0, 0.0])
        graph.embedding_store.add(('person', 2), 'off-topic', vector=[0.0, 1.0, 0.0, 0.0])
        graph.embedding_store.add(('person', 3), 'on-topic', vector=[0.8, 0.6, 0.0, 0.0])

        plain = {r.node_ref: r.score for r in graph.semantic_multi_hop(
            query, rel_type='KNOWS', max_hops=1, threshold=0.5, top_k_seeds=1)}
        # Without blending both neighbours score identically on distance alone.
        assert plain[('person', 2)] == pytest.approx(plain[('person', 3)])

        blended = {r.node_ref: r.score for r in graph.semantic_multi_hop(
            query, rel_type='KNOWS', max_hops=1, threshold=0.5, top_k_seeds=1,
            blend=0.5)}
        assert blended[('person', 3)] > blended[('person', 2)]
        # A seed is unaffected by blending: its own score is its base.
        assert blended[('person', 1)] == pytest.approx(plain[('person', 1)])

    def test_blend_out_of_range_rejected(self, encoder):
        graph = self._two_node_graph(encoder)
        with pytest.raises(ValueError):
            graph.semantic_multi_hop([1.0, 0.0, 0.0, 0.0], blend=1.5)

    def test_negative_seed_does_not_grow_with_distance(self, encoder):
        """Decay used to raise the score of a negatively-matching seed."""
        graph = SemanticGraph(name='neg', embedding_db=':memory:',
                              encoder=encoder, auto_embed=False)
        graph.add_node('person', 1, name='A')
        graph.add_node('person', 2, name='B')
        graph.add_edge('KNOWS', ('person', 1), ('person', 2))
        graph.embedding_store.add(('person', 1), 'a', vector=[-1.0, 0.0, 0.0, 0.0])
        graph.embedding_store.add(('person', 2), 'b', vector=[0.0, 0.0, 1.0, 0.0])

        results = {r.node_ref: r for r in graph.semantic_multi_hop(
            [1.0, 0.0, 0.0, 0.0], rel_type='KNOWS', max_hops=1, threshold=-1.0)}
        assert results[('person', 1)].score == pytest.approx(-1.0)
        # 0.0, not -1.0 * 0.8 = -0.8, which would have outranked the seed.
        assert results[('person', 2)].score == pytest.approx(0.0)


class TestThreadSafety:
    """A shared sqlite connection needs a lock, not just check_same_thread."""

    def test_concurrent_reads_and_writes(self, encoder):
        store = EmbeddingStore(db_path=':memory:', encoder=encoder)
        threads, ops = 8, 150
        errors = []
        writes_per_thread = len([i for i in range(ops) if i % 3 == 0])

        def worker(tid):
            for i in range(ops):
                try:
                    if i % 3 == 0:
                        store.add(('n', tid * ops + i), f'text {tid}-{i}')
                    elif i % 3 == 1:
                        store.similarity_search(f'query {i}', top_k=5)
                    else:
                        store.count()
                except Exception as exc:  # noqa: BLE001 - recording, not handling
                    errors.append(f'{type(exc).__name__}: {exc}')

        workers = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join()

        assert errors == []
        assert store.count() == threads * writes_per_thread
        store.close()


class TestDimensionValidation:
    """Mismatched vectors must fail loudly, not score plausible nonsense."""

    def test_short_vector_rejected(self, embedding_store):
        embedding_store.add(('a', 1), 'full width')
        with pytest.raises(DimensionMismatchError):
            embedding_store.add(('a', 2), 'too short', vector=[1.0, 0.0])

    def test_empty_vector_rejected(self, embedding_store):
        with pytest.raises(DimensionMismatchError):
            embedding_store.add(('a', 1), 'empty', vector=[])

    def test_query_dimension_checked(self, embedding_store):
        embedding_store.add(('a', 1), 'full width')
        with pytest.raises(DimensionMismatchError):
            embedding_store.similarity_search([1.0, 0.0], top_k=1)

    def test_dimension_pinned_by_first_vector(self):
        store = EmbeddingStore(db_path=':memory:')
        assert store.dimension is None
        store.add(('a', 1), 'x', vector=[1.0, 0.0, 0.0])
        assert store.dimension == 3

    def test_text_without_encoder_rejected(self):
        store = EmbeddingStore(db_path=':memory:')
        with pytest.raises(NoEncoderError):
            store.add(('a', 1), 'no encoder here')


class TestEmbeddingPersistence:
    """Embeddings must survive a save/load round trip."""

    def test_save_and_load_restores_embeddings(self, populated_graph, encoder):
        with tempfile.NamedTemporaryFile(suffix='.isong', delete=False) as f:
            graph_path = f.name
        sidecar = SemanticGraph.embeddings_path_for(graph_path)

        try:
            populated_graph.save(graph_path)
            assert sidecar.is_file()

            loaded = SemanticGraph.load(graph_path, encoder=encoder)
            assert loaded.node_count() == populated_graph.node_count()
            assert loaded.edge_count() == populated_graph.edge_count()
            assert loaded.embedding_store.count() == populated_graph.embedding_store.count()

            before = populated_graph.get_embedding(('person', 1))
            after = loaded.get_embedding(('person', 1))
            assert after is not None
            assert after.text == before.text
            assert after.vector == pytest.approx(before.vector)

            # and search actually works on the loaded graph
            assert loaded.similarity_search('software code', top_k=3, threshold=-1.0)
        finally:
            Path(graph_path).unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)

    def test_save_can_skip_embeddings(self, populated_graph):
        with tempfile.NamedTemporaryFile(suffix='.isong', delete=False) as f:
            graph_path = f.name
        sidecar = SemanticGraph.embeddings_path_for(graph_path)
        try:
            populated_graph.save(graph_path, embeddings=False)
            assert not sidecar.exists()
        finally:
            Path(graph_path).unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)

    def test_load_accepts_an_encoder(self, populated_graph, encoder):
        """Without this the first search on a loaded graph downloads a model."""
        with tempfile.NamedTemporaryFile(suffix='.isong', delete=False) as f:
            graph_path = f.name
        sidecar = SemanticGraph.embeddings_path_for(graph_path)
        try:
            populated_graph.save(graph_path)
            loaded = SemanticGraph.load(graph_path, encoder=encoder)
            assert loaded.encoder is encoder
        finally:
            Path(graph_path).unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)

    def test_export_import_round_trip(self, embedding_store, encoder):
        embedding_store.add(('person', 1), 'Alice')
        embedding_store.add(('account', '007'), 'Bond')
        payload = embedding_store.export_embeddings()

        assert payload['format'] == EMBEDDING_FORMAT
        assert payload['count'] == 2

        other = EmbeddingStore(db_path=':memory:', encoder=encoder)
        assert other.import_embeddings(payload) == 2
        # int and str ids keep their original type across the round trip
        assert other.get(('person', 1)).text == 'Alice'
        assert other.get(('account', '007')).text == 'Bond'
        assert other.get(('account', '007')).node_ref == ('account', '007')

    def test_import_rejects_foreign_payload(self, embedding_store):
        with pytest.raises(Exception):
            embedding_store.import_embeddings({'format': 'something-else'})


class TestSemanticGraphExtras:
    """Capabilities added on top of the base traversal."""

    def test_similar_to_node_excludes_itself(self, populated_graph):
        results = populated_graph.similar_to_node(('person', 1), top_k=3,
                                                  threshold=-1.0)
        assert results
        assert all(r.node_ref != ('person', 1) for r in results)
        assert len(results) <= 3

    def test_similar_to_node_without_embedding(self, semantic_graph):
        semantic_graph.add_node('person', 99, name='unembedded')
        assert semantic_graph.similar_to_node(('person', 99)) == []

    def test_semantic_subgraph_is_a_working_graph(self, populated_graph):
        sub = populated_graph.semantic_subgraph(
            query='software engineer', rel_type='KNOWS', max_hops=1,
            top_k_seeds=1, top_k_results=3, threshold=-1.0, name='slice'
        )
        assert isinstance(sub, SemanticGraph)
        assert sub.name == 'slice'
        assert 0 < sub.node_count() <= 3
        # every node kept its embedding
        assert sub.embedding_store.count() == sub.node_count()
        # edges are induced: both endpoints are present
        for edge in sub.edges():
            assert sub.get_node_by_ref(edge.source) is not None
            assert sub.get_node_by_ref(edge.target) is not None

    def test_embed_all_nodes_batches(self, encoder):
        graph = SemanticGraph(name='batch', embedding_db=':memory:',
                              encoder=encoder, auto_embed=False)
        for i in range(10):
            graph.add_node('doc', i, name=f'Document {i}')
        assert graph.embed_all_nodes(batch_size=3) == 10
        assert graph.embedding_store.count() == 10

    def test_embed_all_nodes_can_skip_existing(self, encoder):
        graph = SemanticGraph(name='skip', embedding_db=':memory:',
                              encoder=encoder, auto_embed=False)
        graph.add_node('doc', 1, name='One')
        graph.add_node('doc', 2, name='Two')
        graph.embed_node(('doc', 1), 'already embedded')
        assert graph.embed_all_nodes(skip_existing=True) == 1
        assert graph.get_embedding(('doc', 1)).text == 'already embedded'

    def test_score_map_covers_every_node(self, populated_graph):
        scores = populated_graph.embedding_store.score_map('anything')
        assert len(scores) == populated_graph.embedding_store.count()

    def test_context_manager_closes_store(self, encoder):
        with SemanticGraph(name='ctx', embedding_db=':memory:',
                           encoder=encoder, auto_embed=False) as graph:
            graph.add_node('person', 1, name='Alice', _embed_text='alice')
            assert graph.embedding_store.count() == 1
        # a closed in-memory store reopens empty rather than raising
        assert graph.embedding_store.count() == 0

    def test_encoder_property_does_not_build_a_model(self, encoder):
        graph = SemanticGraph(name='lazy', embedding_db=':memory:')
        assert graph.encoder is None
        graph.set_encoder(encoder)
        assert graph.encoder is encoder


class TestPropertyRendering:
    """Typed property values must render the same way in every port."""

    def test_embed_text_is_identical_across_ports(self, encoder):
        """The same golden string is asserted in all six suites.

        ISONGraph 1.4.0 made property values typed, which is where the ports
        could drift: Python would otherwise render None as "None" and True as
        "True", while JavaScript renders "null" and "true". Different text
        means a different vector, which would break the cross-port guarantee.
        """
        graph = SemanticGraph(name='props', embedding_db=':memory:', encoder=encoder)
        graph.add_node('doc', '1', name='Report', description=None,
                       content=1.5, text=True)

        assert graph.get_embedding(('doc', '1')).text == 'Report 1.5 true'

    def test_null_contributes_nothing(self, encoder):
        graph = SemanticGraph(name='props', embedding_db=':memory:', encoder=encoder)
        assert graph.embed_text_for({'name': None, 'description': 'kept'}) == 'kept'

    def test_booleans_use_the_ison_spelling(self, encoder):
        graph = SemanticGraph(name='props', embedding_db=':memory:', encoder=encoder)
        assert graph.property_text(True) == 'true'
        assert graph.property_text(False) == 'false'


class TestMockEncoderQuality:
    """MockEncoder must be collision-free and identical across ports."""

    def test_no_collisions_across_many_texts(self):
        enc = MockEncoder(dimension=32)
        seen = {tuple(enc.encode(f'text{i}')) for i in range(5000)}
        assert len(seen) == 5000

    def test_cross_language_golden_vector(self):
        """The same values are asserted in the TS, JS, Rust and C++ suites.

        Keeping the ports bit-identical is what lets an embedding store
        written by one be searched by another.
        """
        assert MockEncoder(4).encode('hello') == pytest.approx(
            [0.8381705284, -0.8255031109, 0.5899617672, 0.4615051746], abs=1e-9
        )
        assert MockEncoder(4).encode('ison') == pytest.approx(
            [-0.3658730984, -0.2765417099, 0.2944871187, -0.6502785683], abs=1e-9
        )
        assert MockEncoder(4).encode('') == pytest.approx(
            [-0.4520676136, -0.3797093630, 0.3339765072, 0.9023951292], abs=1e-9
        )

    def test_values_are_in_range(self):
        for value in MockEncoder(128).encode('range check'):
            assert -1.0 <= value < 1.0

    def test_rejects_bad_dimension(self):
        with pytest.raises(ValueError):
            MockEncoder(0)


# =============================================================================
# sqlite-vec backend
# =============================================================================

sqlite_vec_installed = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None,
    reason="sqlite-vec is not installed (pip install 'isongraph-vector[vec]')",
)


@sqlite_vec_installed
class TestSqliteVecBackend:
    """The vec0 index must agree with the scan it replaces, exactly."""

    @staticmethod
    def _pair(encoder, n=200, dim=32):
        """Two stores holding identical vectors, one scanning and one indexed."""
        rng = random.Random(11)
        entries = [
            (('n', i), f'text {i}', [rng.uniform(-1, 1) for _ in range(dim)])
            for i in range(n)
        ]
        scan = EmbeddingStore(db_path=':memory:', encoder=encoder)
        indexed = EmbeddingStore(db_path=':memory:', encoder=encoder, sqlite_vec=True)
        scan.add_batch(entries)
        indexed.add_batch(entries)
        return scan, indexed, [e[2] for e in entries]

    def test_flag_reports_the_backend(self, encoder):
        plain = EmbeddingStore(db_path=':memory:', encoder=encoder)
        indexed = EmbeddingStore(db_path=':memory:', encoder=encoder, sqlite_vec=True)
        assert plain.sqlite_vec is False
        assert indexed.sqlite_vec is True
        plain.close()
        indexed.close()

    def test_matches_the_scan_exactly(self, encoder):
        scan, indexed, vectors = self._pair(encoder)
        try:
            for i in (0, 17, 99, 150):
                want = scan.similarity_search(vectors[i], top_k=10, threshold=-1.0)
                got = indexed.similarity_search(vectors[i], top_k=10, threshold=-1.0)
                assert [r.node_ref for r in got] == [r.node_ref for r in want]
                for a, b in zip(want, got):
                    assert a.score == pytest.approx(b.score, abs=1e-5)
                    assert a.text == b.text
        finally:
            scan.close()
            indexed.close()

    def test_node_type_filter_stays_exact(self, encoder):
        """node_type is a vec0 metadata column, not a post-filter.

        Filtering after a top-k fetch would silently drop matches whenever the
        first k hits were all of the wrong type.
        """
        store = EmbeddingStore(db_path=':memory:', encoder=encoder, sqlite_vec=True)
        try:
            for i in range(50):
                store.add(('crowd', i), f'crowd {i}')
            store.add(('rare', 1), 'the only rare node')

            results = store.similarity_search('anything', top_k=5,
                                              node_type='rare', threshold=-1.0)
            assert len(results) == 1
            assert results[0].node_ref == ('rare', 1)
        finally:
            store.close()

    def test_updates_and_deletes_reach_the_index(self, encoder):
        store = EmbeddingStore(db_path=':memory:', encoder=encoder, sqlite_vec=True)
        try:
            store.add(('a', 1), 'first text')
            store.add(('a', 2), 'second text')

            # overwrite: the index must return the new vector, not the old one
            store.add(('a', 1), 'first text', vector=store.get(('a', 2)).vector)
            top = store.similarity_search(store.get(('a', 2)).vector, top_k=2,
                                          threshold=-1.0)
            assert {r.node_ref for r in top} == {('a', 1), ('a', 2)}
            assert all(r.score == pytest.approx(1.0, abs=1e-5) for r in top)

            store.remove(('a', 1))
            after = store.similarity_search('anything', top_k=10, threshold=-1.0)
            assert [r.node_ref for r in after] == [('a', 2)]

            store.clear()
            assert store.similarity_search('anything', top_k=10, threshold=-1.0) == []
        finally:
            store.close()

    def test_indexes_a_database_written_without_it(self, encoder):
        """Opening an existing store with sqlite_vec=True backfills the index."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            plain = EmbeddingStore(db_path=db_path, encoder=encoder)
            for i in range(20):
                plain.add(('n', i), f'text {i}')
            probe = plain.get(('n', 7)).vector
            plain.close()

            indexed = EmbeddingStore(db_path=db_path, encoder=encoder, sqlite_vec=True)
            try:
                assert indexed.count() == 20
                results = indexed.similarity_search(probe, top_k=3, threshold=-1.0)
                assert results[0].node_ref == ('n', 7)
                assert results[0].score == pytest.approx(1.0, abs=1e-5)
            finally:
                indexed.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_import_populates_the_index(self, encoder):
        source = EmbeddingStore(db_path=':memory:', encoder=encoder)
        target = EmbeddingStore(db_path=':memory:', encoder=encoder, sqlite_vec=True)
        try:
            source.add(('person', 1), 'Alice')
            source.add(('account', '007'), 'Bond')
            assert target.import_embeddings(source.export_embeddings()) == 2

            probe = source.get(('person', 1)).vector
            results = target.similarity_search(probe, top_k=2, threshold=-1.0)
            assert results[0].node_ref == ('person', 1)
        finally:
            source.close()
            target.close()

    def test_score_map_still_covers_everything(self, encoder):
        """Blending needs every score, so score_map must not take the top-k path."""
        store = EmbeddingStore(db_path=':memory:', encoder=encoder, sqlite_vec=True)
        try:
            for i in range(30):
                store.add(('n', i), f'text {i}')
            assert len(store.score_map('anything')) == 30
        finally:
            store.close()

    def test_top_k_none_returns_everything(self, encoder):
        store = EmbeddingStore(db_path=':memory:', encoder=encoder, sqlite_vec=True)
        try:
            for i in range(12):
                store.add(('n', i), f'text {i}')
            assert len(store.similarity_search('anything', top_k=None, threshold=-1.0)) == 12
        finally:
            store.close()

    def test_graph_search_agrees_with_the_scan(self, encoder):
        """semantic_multi_hop seeds from the index when blend is 0."""
        def build(use_vec):
            graph = SemanticGraph(name='vec', embedding_db=':memory:', encoder=encoder,
                                  auto_embed=False, sqlite_vec=use_vec)
            for i in range(1, 6):
                graph.add_node('person', i, name=f'P{i}')
                graph.embed_node(('person', i), f'person number {i}')
            for i in range(1, 5):
                graph.add_edge('KNOWS', ('person', i), ('person', i + 1))
            return graph

        scan, indexed = build(False), build(True)
        try:
            for blend in (0.0, 0.5):
                a = scan.semantic_multi_hop('person number 2', rel_type='KNOWS',
                                            max_hops=2, threshold=-1.0, blend=blend)
                b = indexed.semantic_multi_hop('person number 2', rel_type='KNOWS',
                                               max_hops=2, threshold=-1.0, blend=blend)
                assert [r.node_ref for r in a] == [r.node_ref for r in b]
                for x, y in zip(a, b):
                    assert x.score == pytest.approx(y.score, abs=1e-5)
                    assert x.hop_count == y.hop_count
        finally:
            scan.close()
            indexed.close()

    def test_zero_query_vector_falls_back_to_the_scan(self, encoder):
        """Cosine is undefined for a zero vector; the scan defines it as 0.0."""
        store = EmbeddingStore(db_path=':memory:', encoder=encoder, sqlite_vec=True)
        try:
            store.add(('a', 1), 'x', vector=[1.0, 0.0, 0.0, 0.0])
            results = store.similarity_search([0.0, 0.0, 0.0, 0.0], top_k=1,
                                              threshold=-1.0)
            assert len(results) == 1
            assert results[0].score == 0.0
        finally:
            store.close()

    def test_survives_a_close_and_reopen(self, encoder):
        """The extension is per-connection, so it must reload on reopen."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            store = EmbeddingStore(db_path=db_path, encoder=encoder, sqlite_vec=True)
            store.add(('a', 1), 'hello')
            probe = store.get(('a', 1)).vector
            store.close()

            results = store.similarity_search(probe, top_k=1, threshold=-1.0)
            assert results[0].node_ref == ('a', 1)
            store.close()
        finally:
            Path(db_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
