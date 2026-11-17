#!/usr/bin/env python3
"""
Tests for isongraph-vector

Uses MockEncoder by default to avoid sentence-transformers dependency in CI.
Set USE_REAL_ENCODER=1 to test with actual SentenceTransformer.
"""

import importlib.util
import os
import random
import sqlite3
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

    def test_semantic_path_prefers_best_scoring_seed(self, encoder):
        """A closer, slightly weaker seed beats a stronger seed far away."""
        g = SemanticGraph(name="p", embedding_db=":memory:", encoder=encoder,
                          auto_embed=False)
        pad = [0.0] * 382
        vectors = {1: [1.0, 0.0] + pad, 2: [0.8, 0.6] + pad, 3: [0.0, 1.0] + pad}
        for i, vec in vectors.items():
            g.add_node('n', i)
            g.embedding_store.add(('n', i), f'node {i}', vector=vec)
        g.add_node('n', 9)
        g.add_node('n', 10)
        g.add_edge('E', ('n', 1), ('n', 9))
        g.add_edge('E', ('n', 9), ('n', 10))
        g.add_edge('E', ('n', 10), ('n', 3))   # seed 1 -> target: 3 hops
        g.add_edge('E', ('n', 2), ('n', 3))    # seed 2 -> target: 1 hop

        result = g.semantic_path([1.0, 0.0] + pad, ('n', 3), top_k_seeds=2, threshold=-1.0,
                                 decay=0.5)
        assert result is not None
        assert result.hop_count == 1
        assert result.path[0] == ('n', 2)

    def test_semantic_path_negative_seed_scores_zero(self, encoder):
        """A negatively matching seed must not gain score from a longer path."""
        g = SemanticGraph(name="p", embedding_db=":memory:", encoder=encoder,
                          auto_embed=False)
        g.add_node('n', 1)
        g.add_node('n', 2)
        g.embed_node(('n', 1), 'x')
        g.embedding_store.add(('n', 1), 'x', vector=[-1.0] + [0.0] * 383)
        g.add_edge('E', ('n', 1), ('n', 2))
        result = g.semantic_path([1.0] + [0.0] * 383, ('n', 2), threshold=-1.0)
        assert result is not None
        assert result.score == 0.0

    def test_update_node_reembeds_changed_embed_field(self, encoder):
        g = SemanticGraph(name="u", embedding_db=":memory:", encoder=encoder,
                          auto_embed=True, embed_fields=['name'])
        g.add_node('p', 1, name='alpha')
        before = g.get_embedding(('p', 1)).vector
        g.update_node('p', 1, name='completely different words')
        record = g.get_embedding(('p', 1))
        assert record.text == 'completely different words'
        assert record.vector != before

    def test_update_node_ignores_non_embed_fields(self, encoder):
        g = SemanticGraph(name="u", embedding_db=":memory:", encoder=encoder,
                          auto_embed=True, embed_fields=['name'])
        g.add_node('p', 1, name='alpha')
        before = g.get_embedding(('p', 1))
        g.update_node('p', 1, age=40)
        assert g.get_embedding(('p', 1)).vector == before.vector

    def test_update_node_clearing_text_drops_embedding(self, encoder):
        g = SemanticGraph(name="u", embedding_db=":memory:", encoder=encoder,
                          auto_embed=True, embed_fields=['name'])
        g.add_node('p', 1, name='alpha')
        g.update_node('p', 1, name=None)
        assert g.get_embedding(('p', 1)) is None

    def test_update_node_respects_auto_embed_off(self, encoder):
        g = SemanticGraph(name="u", embedding_db=":memory:", encoder=encoder,
                          auto_embed=False, embed_fields=['name'])
        g.add_node('p', 1, name='alpha')
        g.embed_node(('p', 1), 'manual text')
        g.update_node('p', 1, name='changed')
        assert g.get_embedding(('p', 1)).text == 'manual text'

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


class TestTopKSelection:
    """Selecting the top k must return exactly what sorting all of them would."""

    def test_matches_a_full_sort_including_ties(self, encoder):
        store = EmbeddingStore(db_path=':memory:', encoder=encoder)
        store.add_batch([(('n', str(i)), f'text {i}') for i in range(500)])
        # deliberate ties: several nodes sharing one vector
        for i in range(5):
            store.add(('tie', str(i)), 'tied', vector=[1.0] + [0.0] * 383)

        for k in (1, 5, 10, 50, 200):
            for qi in range(5):
                query = encoder.encode(f'text {qi * 17}')
                got = store.similarity_search(query, top_k=k, threshold=-1.0)
                everything = store.similarity_search(query, top_k=None, threshold=-1.0)
                assert [(r.node_ref, r.score) for r in got] ==                        [(r.node_ref, r.score) for r in everything[:k]], f'k={k}'
        store.close()

    def test_top_k_larger_than_the_store(self, encoder):
        store = EmbeddingStore(db_path=':memory:', encoder=encoder)
        store.add_batch([(('n', str(i)), f'text {i}') for i in range(3)])
        results = store.similarity_search('anything', top_k=100, threshold=-1.0)
        assert len(results) == 3
        assert results == sorted(results, key=lambda r: r.score, reverse=True)
        store.close()

    def test_top_k_zero_returns_nothing(self, encoder):
        store = EmbeddingStore(db_path=':memory:', encoder=encoder)
        store.add(('n', '1'), 'one')
        assert store.similarity_search('anything', top_k=0, threshold=-1.0) == []
        store.close()


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

    def test_round_trip_preserves_properties_and_edges(self, encoder):
        """load() rebuilds through the public API, so nothing may be lost.

        It used to copy ISONGraph's private attributes across wholesale.
        Going through add_node/add_edge instead means the round trip has to
        actually carry node properties, edge properties and their types.
        """
        graph = SemanticGraph(name='rt', embedding_db=':memory:',
                              encoder=encoder, auto_embed=False)
        graph.add_node('person', 1, name='Alice', age=30, active=True)
        graph.add_node('person', 2, name='Bob')
        graph.add_edge('KNOWS', ('person', 1), ('person', 2),
                       since=2020, strength=0.75)
        graph.embed_node(('person', 1), 'alice text')

        with tempfile.NamedTemporaryFile(suffix='.isong', delete=False) as f:
            graph_path = f.name
        sidecar = SemanticGraph.embeddings_path_for(graph_path)
        try:
            graph.save(graph_path)
            loaded = SemanticGraph.load(graph_path, encoder=encoder)

            assert loaded.node_count() == 2
            assert loaded.edge_count() == 1

            node = loaded.get_node_by_ref(('person', 1))
            assert node.properties == {'name': 'Alice', 'age': 30, 'active': True}

            edge = next(iter(loaded.edges('KNOWS')))
            assert edge.properties == {'since': 2020, 'strength': 0.75}

            # the rebuilt indexes actually work
            assert loaded.neighbors(('person', 1), 'KNOWS') == [('person', 2)]
            assert loaded.get_embedding(('person', 1)).text == 'alice text'
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

    def test_floats_render_the_same_way_in_every_port(self):
        """The same values are asserted in all six suites.

        Two ports used to disagree here. Python alone wrote an integral
        float as "1.0" where the others write "1", and C++ used the
        default six significant figures, rendering 1/3 as "0.333333".
        Either one means a different embedding text, and so a different
        vector, for the same graph.
        """
        text = SemanticGraph.property_text
        assert text(1.0) == '1'
        assert text(100.0) == '100'
        assert text(1.5) == '1.5'
        assert text(0.1) == '0.1'
        assert text(1 / 3) == '0.3333333333333333'


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

def _can_load_sqlite_extensions() -> bool:
    """Whether this interpreter can load a SQLite extension at all.

    CPython only exposes enable_load_extension when it was configured with
    --enable-loadable-sqlite-extensions, and several common builds are not:
    the macOS builds from python.org and actions/setup-python among them. The
    package being installed therefore does not mean vec0 can be loaded, and
    the store raises EmbeddingError rather than pretending otherwise.
    """
    try:
        with sqlite3.connect(":memory:") as conn:
            return hasattr(conn, "enable_load_extension")
    except sqlite3.Error:  # pragma: no cover - a broken sqlite3 build
        return False


sqlite_vec_installed = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None
    or not _can_load_sqlite_extensions(),
    reason="needs sqlite-vec (pip install 'isongraph-vector[vec]') and a "
           "Python built with --enable-loadable-sqlite-extensions",
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


# =============================================================================
# Graph / vector consistency, model lifecycle, connectivity, weights, filters
# =============================================================================

import json
import warnings

import isongraph_vector as _iv
from isongraph_vector import (
    GraphEmbeddingMismatchError,
    ModelMismatchError,
    graph_digest,
)


def _named_graph(encoder, names=('Alice', 'Bob', 'Carol')):
    """Auto-embedding graph whose vectors derive from node names."""
    graph = SemanticGraph(name='g', embedding_db=':memory:', encoder=encoder,
                          auto_embed=True)
    for i, name in enumerate(names, start=1):
        graph.add_node('person', i, name=name)
    graph.add_edge('KNOWS', ('person', 1), ('person', 2))
    return graph


def _vector_graph():
    """
    Hand-vectored graph for traversal tests (query [1,0,0,0] is the target).

        person 1  --KNOWS--> company 100 --KNOWS--> person 3
        person 2  (isolated)                         person 3 has no vector

    company 100 is the single best match for the query, person 1 next.
    """
    graph = SemanticGraph(name='v', embedding_db=':memory:', auto_embed=False)
    graph.add_node('person', 1, name='P1', active=True)
    graph.add_node('person', 2, name='P2', active=False)
    graph.add_node('person', 3, name='P3', active=True)
    graph.add_node('company', 100, name='C')
    graph.add_edge('KNOWS', ('person', 1), ('company', 100))
    graph.add_edge('KNOWS', ('company', 100), ('person', 3))
    store = graph.embedding_store
    store.add(('company', 100), 'c', vector=[1.0, 0.0, 0.0, 0.0])
    store.add(('person', 1), 'p1', vector=[0.9, 0.1, 0.0, 0.0])
    store.add(('person', 2), 'p2', vector=[0.8, 0.2, 0.0, 0.0])
    return graph


Q = [1.0, 0.0, 0.0, 0.0]


class TestSaveLoadConsistency:
    """The graph file and its embeddings sidecar must stay a matched pair."""

    def test_sidecar_records_digest_and_stays_version_1(self, encoder, tmp_path):
        graph = _named_graph(encoder)
        path = tmp_path / 'g.isong'
        graph.save(path)
        payload = json.loads(SemanticGraph.embeddings_path_for(path).read_text('utf-8'))

        assert payload['format'] == EMBEDDING_FORMAT
        assert payload['version'] == 1
        assert payload['graph_digest'].startswith('sha256:')
        assert payload['graph_digest_basis'] == _iv.GRAPH_DIGEST_BASIS
        assert payload['encoder_model'] == encoder.model_name
        # Optional metadata trails the array so naive readers (the C++ one
        # scans for keys) meet "format"/"version"/"embeddings" first.
        keys = list(payload)
        assert keys[:5] == ['format', 'version', 'dimension', 'count', 'embeddings']

    def test_saves_leave_no_temp_files(self, encoder, tmp_path):
        _named_graph(encoder).save(tmp_path / 'g.isong')
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            'g.isong', 'g.isong.embeddings.json']

    def test_failed_write_keeps_the_previous_files(self, encoder, tmp_path, monkeypatch):
        path = tmp_path / 'g.isong'
        graph = _named_graph(encoder)
        graph.save(path)
        before = path.read_text('utf-8')
        sidecar_before = SemanticGraph.embeddings_path_for(path).read_text('utf-8')

        graph.add_node('person', 9, name='Zed')

        def boom(src, dst):
            raise OSError('disk full')

        monkeypatch.setattr(_iv.os, 'replace', boom)
        with pytest.raises(OSError):
            graph.save(path)
        monkeypatch.undo()

        assert path.read_text('utf-8') == before
        assert SemanticGraph.embeddings_path_for(path).read_text('utf-8') == sidecar_before
        assert [p.name for p in tmp_path.iterdir() if p.name.endswith('.tmp')] == []

    def test_matching_pair_loads_clean(self, encoder, tmp_path):
        """Typed values must not make a healthy save look mismatched."""
        graph = SemanticGraph(name='t', embedding_db=':memory:', encoder=encoder,
                              auto_embed=False)
        graph.add_node('person', 1, name='A b', age=30, ratio=1.0, ok=True, note=None)
        graph.add_node('account', '007', name='Bond')
        graph.add_edge('KNOWS', ('person', 1), ('account', '007'), w=0.5, n=2)
        graph.embed_node(('person', 1), 'a')
        path = tmp_path / 't.isong'
        graph.save(path)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            loaded = SemanticGraph.load(path, encoder=encoder)
        assert loaded.embedding_store.count() == 1

    def test_isonl_round_trip_verifies(self, encoder, tmp_path):
        graph = _named_graph(encoder)
        path = tmp_path / 'g.isonl'
        graph.save(path)
        loaded = SemanticGraph.load(path, encoder=encoder)
        assert loaded.embedding_store.count() == 3

    def test_sidecar_from_another_save_is_rejected(self, encoder, tmp_path):
        a = _named_graph(encoder, ('Alice', 'Bob', 'Carol'))
        b = _named_graph(encoder, ('Xavier', 'Yolanda', 'Zed'))
        pa, pb = tmp_path / 'a.isong', tmp_path / 'b.isong'
        a.save(pa)
        b.save(pb)
        # b's vectors dropped next to a's graph
        SemanticGraph.embeddings_path_for(pa).write_text(
            SemanticGraph.embeddings_path_for(pb).read_text('utf-8'), encoding='utf-8')
        with pytest.raises(GraphEmbeddingMismatchError, match='digest'):
            SemanticGraph.load(pa, encoder=encoder)

    def test_edited_graph_is_rejected_by_default(self, encoder, tmp_path):
        path = tmp_path / 'g.isong'
        _named_graph(encoder).save(path)
        path.write_text(path.read_text('utf-8').replace('Bob', 'Robert'), encoding='utf-8')
        with pytest.raises(GraphEmbeddingMismatchError):
            SemanticGraph.load(path, encoder=encoder)

    def test_warn_mode_loads_the_old_vectors_anyway(self, encoder, tmp_path):
        path = tmp_path / 'g.isong'
        _named_graph(encoder).save(path)
        path.write_text(path.read_text('utf-8').replace('Bob', 'Robert'), encoding='utf-8')
        with pytest.warns(UserWarning, match='digest'):
            loaded = SemanticGraph.load(path, encoder=encoder, on_mismatch='warn')
        assert loaded.get_embedding(('person', 2)).text == 'Bob'  # the stale text

    def test_reembed_mode_rebuilds_from_the_edited_graph(self, encoder, tmp_path):
        path = tmp_path / 'g.isong'
        _named_graph(encoder).save(path)
        path.write_text(path.read_text('utf-8').replace('Bob', 'Robert'), encoding='utf-8')
        loaded = SemanticGraph.load(path, encoder=encoder, on_mismatch='reembed')
        record = loaded.get_embedding(('person', 2))
        assert record.text == 'Robert'
        assert record.vector == pytest.approx(encoder.encode('Robert'), abs=1e-6)
        assert loaded.embedding_store.count() == 3

    def test_sidecar_without_digest_loads_as_before(self, encoder, tmp_path):
        path = tmp_path / 'g.isong'
        _named_graph(encoder).save(path)
        sidecar = SemanticGraph.embeddings_path_for(path)
        payload = json.loads(sidecar.read_text('utf-8'))
        for key in ('graph_digest', 'graph_digest_basis', 'encoder_model'):
            payload.pop(key, None)
        sidecar.write_text(json.dumps(payload), encoding='utf-8')
        # even an edited graph is accepted: there is nothing to compare with
        path.write_text(path.read_text('utf-8').replace('Bob', 'Robert'), encoding='utf-8')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            loaded = SemanticGraph.load(path, encoder=encoder)
        assert loaded.embedding_store.count() == 3

    def test_digest_of_unknown_basis_is_not_compared(self, encoder, tmp_path):
        path = tmp_path / 'g.isong'
        _named_graph(encoder).save(path)
        sidecar = SemanticGraph.embeddings_path_for(path)
        payload = json.loads(sidecar.read_text('utf-8'))
        payload['graph_digest'] = 'sha256:' + '0' * 64
        payload['graph_digest_basis'] = 'some-other-port-v9'
        sidecar.write_text(json.dumps(payload), encoding='utf-8')
        assert SemanticGraph.load(path, encoder=encoder).embedding_store.count() == 3

    def test_legacy_payload_with_unknown_fields_imports(self, encoder):
        """Version-1 readers must ignore fields they do not know."""
        source = EmbeddingStore(db_path=':memory:', encoder=encoder)
        source.add(('person', 1), 'Alice')
        payload = source.export_embeddings(graph_digest='sha256:abc')
        payload['some_future_field'] = {'x': 1}
        target = EmbeddingStore(db_path=':memory:', encoder=encoder)
        assert target.import_embeddings(payload) == 1
        assert target.last_import_meta['graph_digest'] == 'sha256:abc'

    def test_rejects_unknown_on_mismatch_mode(self, encoder, tmp_path):
        path = tmp_path / 'g.isong'
        _named_graph(encoder).save(path)
        with pytest.raises(ValueError, match='on_mismatch'):
            SemanticGraph.load(path, encoder=encoder, on_mismatch='ignore')

    def test_digest_ignores_insertion_order_but_sees_content(self, encoder):
        a = SemanticGraph(auto_embed=False)
        b = SemanticGraph(auto_embed=False)
        a.add_node('t', 1, x=1)
        a.add_node('t', 2, x=2)
        a.add_edge('R', ('t', 1), ('t', 2))
        b.add_node('t', 2, x=2)
        b.add_node('t', 1, x=1)
        b.add_edge('R', ('t', 1), ('t', 2))
        assert graph_digest(a) == graph_digest(b)
        b.update_node('t', 1, x=99)
        assert graph_digest(a) != graph_digest(b)
        # an int id and a str id that print alike are different nodes
        c = SemanticGraph(auto_embed=False)
        d = SemanticGraph(auto_embed=False)
        c.add_node('t', 7)
        d.add_node('t', '7')
        assert graph_digest(c) != graph_digest(d)


class TestModelLifecycle:
    """Vectors from one model must not be silently searched with another."""

    @staticmethod
    def _retagged(dimension, name):
        enc = MockEncoder(dimension)
        enc.model_name = name
        return enc

    def test_models_counts_vectors_per_model(self, encoder):
        graph = _named_graph(encoder)
        assert graph.embedding_store.models() == {encoder.model_name: 3}

    def test_stale_by_model_name_then_reembed(self, encoder):
        graph = _named_graph(MockEncoder(16))
        graph.set_encoder(self._retagged(16, 'next-gen-16'))

        stale = graph.stale_embeddings()
        assert sorted(stale) == [('person', 1), ('person', 2), ('person', 3)]
        assert graph.embedding_store.models() == {'mock-16': 3}

        assert graph.reembed_stale() == 3
        assert graph.stale_embeddings() == []
        assert graph.embedding_store.models() == {'next-gen-16': 3}

    def test_nothing_is_stale_under_the_same_encoder(self, encoder):
        graph = _named_graph(encoder)
        assert graph.stale_embeddings() == []
        assert graph.reembed_stale() == 0

    def test_stale_by_dimension_resets_the_store(self):
        graph = _named_graph(MockEncoder(8))
        graph.set_encoder(MockEncoder(16))
        assert len(graph.stale_embeddings()) == 3

        assert graph.reembed_stale() == 3
        assert graph.embedding_store.dimension == 16
        assert len(graph.get_embedding(('person', 1)).vector) == 16
        assert graph.stale_embeddings() == []

    def test_reembed_keeps_custom_embed_text(self):
        graph = SemanticGraph(embedding_db=':memory:', encoder=MockEncoder(8),
                              auto_embed=False)
        graph.add_node('person', 1, name='A', _embed_text='a very specific blurb')
        graph.set_encoder(self._retagged(8, 'v2'))
        graph.reembed_stale()
        record = graph.get_embedding(('person', 1))
        assert record.text == 'a very specific blurb'
        assert record.model == 'v2'

    def test_unknown_model_is_not_stale_unless_asked(self):
        graph = SemanticGraph(embedding_db=':memory:', encoder=MockEncoder(8),
                              auto_embed=False)
        graph.add_node('person', 1)
        graph.embedding_store.encoder = None          # a vector handed in -> "provided"
        graph.embedding_store.add(('person', 1), 'x', vector=[0.1] * 8)
        graph.embedding_store.encoder = graph.encoder
        assert graph.embedding_store.models() == {'provided': 1}
        assert graph.stale_embeddings() == []
        assert graph.stale_embeddings(include_unknown=True) == [('person', 1)]

    def test_text_search_with_a_different_model_warns_once(self):
        graph = _named_graph(MockEncoder(16))
        graph.set_encoder(self._retagged(16, 'other-16'))
        with pytest.warns(UserWarning, match='other-16'):
            graph.similarity_search('hello', top_k=1, threshold=-1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            graph.similarity_search('hello again', top_k=1, threshold=-1.0)

    def test_text_search_with_a_different_model_can_raise(self):
        graph = SemanticGraph(embedding_db=':memory:', encoder=MockEncoder(16),
                              auto_embed=True, on_model_mismatch='error')
        graph.add_node('person', 1, name='Alice')
        graph.set_encoder(self._retagged(16, 'other-16'))
        with pytest.raises(ModelMismatchError, match='mock-16'):
            graph.similarity_search('hello', top_k=1)
        with pytest.raises(ModelMismatchError):
            graph.semantic_multi_hop('hello')
        # re-embedding is the cure
        graph.reembed_stale()
        assert graph.similarity_search('hello', top_k=1, threshold=-1.0)

    def test_vector_queries_and_ignore_mode_are_never_checked(self):
        graph = _named_graph(MockEncoder(16))
        graph.set_encoder(self._retagged(16, 'other-16'))
        probe = MockEncoder(16).encode('x')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            graph.similarity_search(probe, top_k=1, threshold=-1.0)
        quiet = SemanticGraph(embedding_db=':memory:', encoder=MockEncoder(16),
                              on_model_mismatch='ignore')
        quiet.add_node('person', 1, name='Alice')
        quiet.set_encoder(self._retagged(16, 'other-16'))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            quiet.similarity_search('x', top_k=1, threshold=-1.0)

    def test_rejects_unknown_mismatch_mode(self):
        with pytest.raises(ValueError):
            EmbeddingStore(db_path=':memory:', on_model_mismatch='loud')

    def test_version_is_recorded_in_the_sidecar(self, tmp_path):
        encoder = MockEncoder(8, version='2026.1')
        graph = _named_graph(encoder)
        path = tmp_path / 'g.isong'
        graph.save(path)
        payload = json.loads(SemanticGraph.embeddings_path_for(path).read_text('utf-8'))
        assert payload['encoder_version'] == '2026.1'
        # an explicit version wins over the encoder's
        store = EmbeddingStore(db_path=':memory:', encoder=encoder, version='explicit')
        store.add(('a', 1), 'x')
        assert store.export_embeddings()['encoder_version'] == 'explicit'
        # and nothing is written when no version is known
        plain = _named_graph(MockEncoder(8))
        assert 'encoder_version' not in plain.embedding_store.export_embeddings()

    def test_loading_with_a_different_version_warns(self, tmp_path):
        graph = _named_graph(MockEncoder(8, version='1'))
        path = tmp_path / 'g.isong'
        graph.save(path)
        with pytest.warns(UserWarning, match="version '1'"):
            SemanticGraph.load(path, encoder=MockEncoder(8, version='2'))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            SemanticGraph.load(path, encoder=MockEncoder(8, version='1'))
            SemanticGraph.load(path, encoder=MockEncoder(8))   # unversioned: no claim


class TestSubgraphConnectivity:
    """semantic_subgraph(connect=True) bridges the selected nodes."""

    @staticmethod
    def _bridge(rel_ab='R', rel_xb='R', a_to_x=True, b_to_x=False):
        graph = SemanticGraph(embedding_db=':memory:', auto_embed=False)
        for ref in ('A', 'B', 'X'):
            graph.add_node('n', ref, kind='hub' if ref == 'X' else 'leaf')
        # a -> x -> b (or b -> x when a_to_x/b_to_x choose the reversed shape)
        if a_to_x:
            graph.add_edge(rel_ab, ('n', 'A'), ('n', 'X'))
        else:
            graph.add_edge(rel_ab, ('n', 'X'), ('n', 'A'))
        if b_to_x:
            graph.add_edge(rel_xb, ('n', 'B'), ('n', 'X'))
        else:
            graph.add_edge(rel_xb, ('n', 'X'), ('n', 'B'))
        graph.embedding_store.add(('n', 'A'), 'a', vector=[1.0, 0.0])
        graph.embedding_store.add(('n', 'B'), 'b', vector=[0.9, 0.1])
        return graph

    @staticmethod
    def _sub(graph, **kw):
        params = dict(max_hops=0, top_k_seeds=2, top_k_results=2, threshold=-1.0)
        params.update(kw)
        return graph.semantic_subgraph([1.0, 0.0], **params)

    @staticmethod
    def _refs(graph):
        return {n.ref for n in graph.nodes()}

    def test_default_leaves_the_selection_disconnected(self):
        sub = self._sub(self._bridge())
        assert self._refs(sub) == {('n', 'A'), ('n', 'B')}
        assert sub.edge_count() == 0

    def test_connect_adds_the_bridging_node_and_its_edges(self):
        sub = self._sub(self._bridge(), connect=True)
        assert self._refs(sub) == {('n', 'A'), ('n', 'B'), ('n', 'X')}
        assert sub.edge_count() == 2
        # the connector carries its own properties
        assert sub.get_node_by_ref(('n', 'X')).properties == {'kind': 'hub'}

    def test_connector_nodes_may_exceed_top_k_results(self):
        sub = self._sub(self._bridge(), connect=True, top_k_results=2)
        assert sub.node_count() == 3

    def test_hop_budget_limits_the_search(self):
        graph = self._bridge()
        assert self._refs(self._sub(graph, connect=True, connect_max_hops=1)) == {
            ('n', 'A'), ('n', 'B')}
        assert ('n', 'X') in self._refs(self._sub(graph, connect=True, connect_max_hops=2))
        with pytest.raises(ValueError):
            self._sub(graph, connect=True, connect_max_hops=0)

    def test_longer_budget_adds_a_chain_of_connectors(self):
        graph = SemanticGraph(embedding_db=':memory:', auto_embed=False)
        for ref in ('A', 'M1', 'M2', 'B'):
            graph.add_node('n', ref)
        graph.add_edge('R', ('n', 'A'), ('n', 'M1'))
        graph.add_edge('R', ('n', 'M1'), ('n', 'M2'))
        graph.add_edge('R', ('n', 'M2'), ('n', 'B'))
        graph.embedding_store.add(('n', 'A'), 'a', vector=[1.0, 0.0])
        graph.embedding_store.add(('n', 'B'), 'b', vector=[0.9, 0.1])
        assert self._sub(graph, connect=True, connect_max_hops=2).node_count() == 2
        assert self._sub(graph, connect=True, connect_max_hops=3).node_count() == 4

    def test_direction_is_respected(self):
        graph = self._bridge(a_to_x=True, b_to_x=True)   # A -> X <- B
        assert self._refs(self._sub(graph, connect=True)) == {('n', 'A'), ('n', 'B')}
        both = self._sub(graph, connect=True, direction=Direction.BOTH)
        assert ('n', 'X') in self._refs(both)

    def test_rel_type_is_respected(self):
        graph = self._bridge(rel_ab='KNOWS', rel_xb='LIKES')
        assert self._refs(self._sub(graph, connect=True, rel_type='KNOWS')) == {
            ('n', 'A'), ('n', 'B')}
        assert ('n', 'X') in self._refs(self._sub(graph, connect=True))

    def test_one_hub_joins_three_components(self):
        graph = SemanticGraph(embedding_db=':memory:', auto_embed=False)
        graph.add_node('n', 'HUB')
        for i, ref in enumerate(('A', 'B', 'C')):
            graph.add_node('n', ref)
            graph.add_edge('R', ('n', ref), ('n', 'HUB'))
            graph.embedding_store.add(('n', ref), ref, vector=[1.0 - i * 0.05, 0.1 * i])
        sub = graph.semantic_subgraph([1.0, 0.0], max_hops=0, top_k_seeds=3,
                                      top_k_results=3, threshold=-1.0,
                                      connect=True, direction=Direction.BOTH)
        assert self._refs(sub) == {('n', 'A'), ('n', 'B'), ('n', 'C'), ('n', 'HUB')}

    def test_strict_filter_bars_failing_nodes_as_connectors(self):
        graph = self._bridge()
        kw = dict(connect=True, node_filter=lambda n: n.properties['kind'] == 'leaf')
        # lax: X fails the filter but may still bridge
        assert ('n', 'X') in self._refs(self._sub(graph, **kw))
        strict = self._sub(graph, strict_filter=True, **kw)
        assert self._refs(strict) == {('n', 'A'), ('n', 'B')}


class TestEdgeWeighting:
    """weight_field scales each hop's decay by an edge property."""

    @staticmethod
    def _fan(p_weight, q_weight='missing'):
        graph = SemanticGraph(embedding_db=':memory:', auto_embed=False)
        for ref in ('S', 'P', 'Q'):
            graph.add_node('n', ref)
        graph.add_edge('R', ('n', 'S'), ('n', 'P'), w=p_weight)
        if q_weight == 'missing':
            graph.add_edge('R', ('n', 'S'), ('n', 'Q'))
        else:
            graph.add_edge('R', ('n', 'S'), ('n', 'Q'), w=q_weight)
        graph.embedding_store.add(('n', 'S'), 's', vector=[1.0, 0.0])
        return graph

    @staticmethod
    def _scores(graph, **kw):
        found = graph.semantic_multi_hop([1.0, 0.0], max_hops=1, top_k_seeds=1,
                                         threshold=-1.0, **kw)
        return {r.node_ref[1]: r.score for r in found}

    def test_default_ignores_edge_properties(self):
        scores = self._scores(self._fan(0.1))
        assert scores['P'] == pytest.approx(0.8)
        assert scores['Q'] == pytest.approx(0.8)

    def test_weight_multiplies_the_decay(self):
        scores = self._scores(self._fan(0.5), weight_field='w')
        assert scores['S'] == pytest.approx(1.0)
        assert scores['P'] == pytest.approx(0.8 * 0.5)
        assert scores['Q'] == pytest.approx(0.8)          # missing -> 1.0

    def test_weights_are_clamped_and_junk_counts_as_one(self):
        assert self._scores(self._fan(5.0), weight_field='w')['P'] == pytest.approx(0.8)
        assert self._scores(self._fan(-3.0), weight_field='w')['P'] == pytest.approx(0.0)
        assert self._scores(self._fan('heavy'), weight_field='w')['P'] == pytest.approx(0.8)
        assert self._scores(self._fan(True), weight_field='w')['P'] == pytest.approx(0.8)

    def test_best_route_can_be_longer_than_the_shortest(self):
        graph = SemanticGraph(embedding_db=':memory:', auto_embed=False)
        for ref in ('S', 'M', 'T'):
            graph.add_node('n', ref)
        graph.add_edge('R', ('n', 'S'), ('n', 'T'), w=0.1)       # direct but weak
        graph.add_edge('R', ('n', 'S'), ('n', 'M'), w=1.0)
        graph.add_edge('R', ('n', 'M'), ('n', 'T'), w=1.0)       # 2 hops, strong
        graph.embedding_store.add(('n', 'S'), 's', vector=[1.0, 0.0])

        plain = {r.node_ref[1]: r for r in graph.semantic_multi_hop(
            [1.0, 0.0], max_hops=2, top_k_seeds=1, threshold=-1.0)}
        assert plain['T'].hop_count == 1
        assert plain['T'].score == pytest.approx(0.8)

        weighted = {r.node_ref[1]: r for r in graph.semantic_multi_hop(
            [1.0, 0.0], max_hops=2, top_k_seeds=1, threshold=-1.0, weight_field='w')}
        assert weighted['T'].hop_count == 2
        assert weighted['T'].path == [('n', 'S'), ('n', 'M'), ('n', 'T')]
        assert weighted['T'].score == pytest.approx(0.64)

    def test_hop_budget_still_applies_to_the_better_route(self):
        graph = SemanticGraph(embedding_db=':memory:', auto_embed=False)
        for ref in ('S', 'M', 'T'):
            graph.add_node('n', ref)
        graph.add_edge('R', ('n', 'S'), ('n', 'T'), w=0.1)
        graph.add_edge('R', ('n', 'S'), ('n', 'M'), w=1.0)
        graph.add_edge('R', ('n', 'M'), ('n', 'T'), w=1.0)
        graph.embedding_store.add(('n', 'S'), 's', vector=[1.0, 0.0])
        found = {r.node_ref[1]: r for r in graph.semantic_multi_hop(
            [1.0, 0.0], max_hops=1, top_k_seeds=1, threshold=-1.0, weight_field='w')}
        assert found['T'].score == pytest.approx(0.8 * 0.1)

    def test_subgraph_selection_follows_the_weights(self):
        graph = self._fan(0.1)            # P weak, Q unweighted
        sub = graph.semantic_subgraph([1.0, 0.0], max_hops=1, top_k_seeds=1,
                                      top_k_results=2, threshold=-1.0,
                                      weight_field='w')
        assert {n.ref[1] for n in sub.nodes()} == {'S', 'Q'}


class TestNodeFiltering:
    """node_filter / node_types gate seeds and results; strict_filter gates paths."""

    def test_similarity_search_filters_before_taking_top_k(self):
        graph = _vector_graph()
        # the single best match is the company; filtering must look past it
        assert graph.similarity_search(Q, top_k=1)[0].node_ref == ('company', 100)
        top = graph.similarity_search(Q, top_k=1, node_types=['person'])
        assert [r.node_ref for r in top] == [('person', 1)]

    def test_node_filter_and_types_combine_with_and(self):
        graph = _vector_graph()
        active = graph.similarity_search(Q, top_k=None, threshold=-1.0,
                                         node_filter=lambda n: n.properties.get('active'))
        assert [r.node_ref for r in active] == [('person', 1)]
        none = graph.similarity_search(Q, top_k=None, threshold=-1.0,
                                       node_types=['company'],
                                       node_filter=lambda n: n.properties.get('active'))
        assert none == []

    def test_a_bare_string_is_one_type_and_empty_matches_nothing(self):
        graph = _vector_graph()
        assert [r.node_ref for r in graph.similarity_search(
            Q, top_k=None, node_types='company')] == [('company', 100)]
        assert graph.similarity_search(Q, top_k=None, node_types=[]) == []

    def test_embeddings_without_a_node_fail_node_filter_only(self):
        graph = _vector_graph()
        graph.embedding_store.add(('ghost', 1), 'g', vector=[1.0, 0.0, 0.0, 0.0])
        assert ('ghost', 1) in {r.node_ref for r in graph.similarity_search(
            Q, top_k=None, threshold=-1.0, node_types=['ghost'])}
        assert ('ghost', 1) not in {r.node_ref for r in graph.similarity_search(
            Q, top_k=None, threshold=-1.0, node_filter=lambda n: True)}

    def test_multi_hop_seeds_and_results_are_filtered(self):
        graph = _vector_graph()
        found = graph.semantic_multi_hop(Q, max_hops=2, top_k_seeds=1,
                                         threshold=-1.0, node_types=['person'])
        refs = [r.node_ref for r in found]
        # the best match (company) did not take the seed slot, and is not returned
        assert refs[0] == ('person', 1)
        assert ('company', 100) not in refs

    def test_filtered_nodes_are_traversed_but_not_returned_by_default(self):
        graph = _vector_graph()
        found = {r.node_ref: r for r in graph.semantic_multi_hop(
            Q, max_hops=2, top_k_seeds=1, threshold=-1.0, node_types=['person'])}
        reach = found[('person', 3)]
        assert reach.path == [('person', 1), ('company', 100), ('person', 3)]
        assert reach.hop_count == 2                    # the company still cost a hop
        assert reach.score == pytest.approx(found[('person', 1)].score * 0.8 ** 2)

    def test_strict_filter_walls_off_the_traversal(self):
        graph = _vector_graph()
        found = {r.node_ref for r in graph.semantic_multi_hop(
            Q, max_hops=2, top_k_seeds=1, threshold=-1.0,
            node_types=['person'], strict_filter=True)}
        assert ('person', 3) not in found
        assert ('company', 100) not in found

    def test_blend_path_respects_the_filter_too(self):
        graph = _vector_graph()
        found = graph.semantic_multi_hop(Q, max_hops=1, top_k_seeds=2, threshold=-1.0,
                                         blend=0.5, node_types=['person'])
        assert found and all(r.node_ref[0] == 'person' for r in found)

    def test_semantic_path_requires_a_passing_target(self):
        graph = _vector_graph()
        assert graph.semantic_path(Q, ('company', 100), threshold=-1.0,
                                   node_types=['person']) is None
        assert graph.semantic_path(Q, ('company', 100), threshold=-1.0) is not None

    def test_semantic_path_may_cross_failing_nodes_unless_strict(self):
        graph = _vector_graph()
        lax = graph.semantic_path(Q, ('person', 3), threshold=-1.0, top_k_seeds=1,
                                  node_types=['person'])
        assert lax.path == [('person', 1), ('company', 100), ('person', 3)]
        strict = graph.semantic_path(Q, ('person', 3), threshold=-1.0, top_k_seeds=1,
                                     node_types=['person'], strict_filter=True)
        assert strict is None

    def test_semantic_path_seeds_come_from_passing_nodes(self):
        graph = _vector_graph()
        result = graph.semantic_path(Q, ('person', 3), threshold=-1.0, top_k_seeds=1,
                                     node_filter=lambda n: n.type == 'person')
        assert result.path[0] == ('person', 1)

    def test_subgraph_holds_only_passing_nodes(self):
        graph = _vector_graph()
        sub = graph.semantic_subgraph(Q, max_hops=2, top_k_seeds=2, threshold=-1.0,
                                      node_types=['person'])
        assert {n.type for n in sub.nodes()} == {'person'}
        assert ('person', 3) in {n.ref for n in sub.nodes()}

    def test_defaults_are_unchanged(self):
        graph = _vector_graph()
        plain = graph.semantic_multi_hop(Q, max_hops=2, top_k_seeds=1, threshold=-1.0)
        assert plain[0].node_ref == ('company', 100)
        assert ('person', 3) in {r.node_ref for r in plain}


@sqlite_vec_installed
class TestSqliteVecLifecycle:
    """The vec0 table is fixed-width, so a dimension change must rebuild it."""

    def test_reembed_to_a_new_dimension_rebuilds_the_index(self):
        graph = SemanticGraph(embedding_db=':memory:', encoder=MockEncoder(8),
                              sqlite_vec=True)
        try:
            graph.add_node('p', 1, name='a')
            graph.add_node('p', 2, name='b')
            graph.set_encoder(MockEncoder(16))
            assert graph.reembed_stale() == 2
            hits = graph.similarity_search('a', top_k=2, threshold=-1.0,
                                           node_types=['p'])
            assert {h.node_ref for h in hits} == {('p', 1), ('p', 2)}
        finally:
            graph.close()


class _RefModel:
    """Brute-force reference for the exact scan: plain dicts, no caching.

    Insertion order mirrors the table's rowid order: an update keeps its
    place, a delete followed by an add goes to the end.
    """

    def __init__(self):
        self.rows = {}          # key -> [ref, vector(float32-rounded), text]

    @staticmethod
    def _key(ref):
        return (str(ref[0]), str(ref[1]))

    @staticmethod
    def _f32(vector):
        import struct
        return list(struct.unpack(f'{len(vector)}f', struct.pack(f'{len(vector)}f', *vector)))

    def put(self, ref, vector, text):
        key = self._key(ref)
        if key in self.rows:
            self.rows[key][1:] = [self._f32(vector), text]
            self.rows[key][0] = ref
        else:
            self.rows[key] = [ref, self._f32(vector), text]

    def drop(self, ref):
        return self.rows.pop(self._key(ref), None) is not None

    def clear(self):
        self.rows.clear()

    @staticmethod
    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

    def search(self, query, top_k=10, node_type=None, threshold=0.0):
        scored = [(self._cos(v, query), ref, text)
                  for ref, v, text in self.rows.values()
                  if (node_type is None or ref[0] == node_type)]
        scored = [t for t in scored if t[0] >= threshold]
        scored.sort(key=lambda t: -t[0])          # stable: ties keep insertion order
        return scored if top_k is None else scored[:max(top_k, 0)]


def _assert_same_ranking(got, expected, tol=1e-5):
    """Same length, scores within tol, and the same ref wherever scores are not tied."""
    assert len(got) == len(expected)
    for i, (r, (score, ref, text)) in enumerate(zip(got, expected)):
        assert r.score == pytest.approx(score, abs=tol), f'rank {i}'
        if r.node_ref != ref:
            # only legitimate when neighbours are within float noise of each other
            near = [e[0] for e in expected if abs(e[0] - score) <= tol]
            assert len(near) > 1, f'rank {i}: {r.node_ref} != {ref}'
        else:
            assert r.text == text


def _cache_snapshot(store):
    cache = store._ensure_cache()
    live = [i for i in range(cache.n) if cache.alive[i]]
    return {
        'refs': [cache.refs[i] for i in live],
        'texts': [cache.texts[i] for i in live],
        'matrix': cache.matrix[live].copy(),
        'norms': cache.norms[live].copy(),
        'types': [cache.refs[i][0] for i in live],
    }


def _assert_cache_matches_rebuild(store):
    """The patched cache must equal one built from scratch from the table."""
    np = pytest.importorskip('numpy')
    patched = _cache_snapshot(store)
    if store._sql:
        store._cache = None                   # force a rebuild from SQLite
        fresh = _cache_snapshot(store)
    else:
        # an in-memory store has no table behind its cache: rebuild a second
        # store from the export instead (every row written once, in order)
        twin = EmbeddingStore(db_path=':memory:')
        twin.import_embeddings(store.export_embeddings())
        fresh = _cache_snapshot(twin)
    assert patched['refs'] == fresh['refs']
    assert patched['texts'] == fresh['texts']
    assert np.array_equal(patched['matrix'], fresh['matrix'])
    assert np.allclose(patched['norms'], fresh['norms'], rtol=1e-6, atol=0)
    # every live row is reachable through the key index, at its own position
    cache = store._cache
    assert len(cache.pos) == cache.n - cache.dead
    for key, i in cache.pos.items():
        assert cache.alive[i] and (str(cache.refs[i][0]), str(cache.refs[i][1])) == key


class TestIncrementalExactSearch:
    """The vectorised exact path and its incrementally patched cache."""

    DIM = 12

    @staticmethod
    def _vec(rng, dim):
        return [rng.uniform(-1, 1) for _ in range(dim)]

    def _check(self, store, model, rng, dim, queries=3):
        for _ in range(queries):
            q = self._vec(rng, dim)
            for top_k in (1, 5, 17, 10_000, None, 0):
                for node_type in (None, 'a', 'b', 'missing'):
                    for threshold in (-1.0, 0.0, 0.35):
                        got = store.similarity_search(
                            q, top_k=top_k, node_type=node_type, threshold=threshold)
                        _assert_same_ranking(
                            got, model.search(q, top_k, node_type, threshold))
            smap = store.score_map(q)
            assert list(smap) == [ref for ref, *_ in model.rows.values()]
            for (ref, v, _), s in zip(model.rows.values(), smap.values()):
                assert s == pytest.approx(model._cos(v, q), abs=1e-5)
            only_b = store.score_map(q, node_type='b')
            assert list(only_b) == [r for r, *_ in model.rows.values() if r[0] == 'b']

    def test_matches_brute_force_through_random_updates_deletes_and_readds(self):
        pytest.importorskip('numpy')
        rng = random.Random(2024)
        dim = self.DIM
        store = EmbeddingStore(db_path=':memory:', dimension=dim)
        model = _RefModel()
        live = []
        next_id = 0

        def new_ref():
            nonlocal next_id
            next_id += 1
            # a mix of int and str ids, and two node types
            return (rng.choice('ab'), next_id if next_id % 3 else f'id{next_id}')

        for _ in range(250):
            ref = new_ref()
            v, text = self._vec(rng, dim), f't{next_id}'
            store.add(ref, text, vector=v)
            model.put(ref, v, text)
            live.append(ref)
        self._check(store, model, rng, dim, queries=1)       # builds the cache

        for step in range(400):
            action = rng.choice(['add', 'add', 'update', 'delete', 'delete', 'readd',
                                 'batch', 'search'])
            if action == 'add':
                ref = new_ref()
                v, text = self._vec(rng, dim), f'n{step}'
                store.add(ref, text, vector=v)
                model.put(ref, v, text)
                live.append(ref)
            elif action == 'update' and live:
                ref = rng.choice(live)
                v, text = self._vec(rng, dim), f'u{step}'
                store.add(ref, text, vector=v)
                model.put(ref, v, text)
            elif action == 'delete' and live:
                ref = live.pop(rng.randrange(len(live)))
                assert store.remove(ref) is True
                assert model.drop(ref) is True
                assert store.remove(ref) is False
            elif action == 'readd' and live:
                ref = live.pop(rng.randrange(len(live)))
                store.remove(ref)
                model.drop(ref)
                v, text = self._vec(rng, dim), f'r{step}'
                store.add(ref, text, vector=v)
                model.put(ref, v, text)
                live.append(ref)
            elif action == 'batch':
                batch = []
                for _ in range(rng.randint(1, 6)):
                    ref = rng.choice(live) if live and rng.random() < 0.4 else new_ref()
                    v, text = self._vec(rng, dim), f'b{step}'
                    batch.append((ref, text, v))
                store.add_batch(batch)
                for ref, text, v in batch:
                    model.put(ref, v, text)
                    if ref not in live:
                        live.append(ref)
            elif action == 'search':
                self._check(store, model, rng, dim, queries=1)
            assert store.count() == len(model.rows)

        self._check(store, model, rng, dim)
        _assert_cache_matches_rebuild(store)
        store.close()

    def test_ties_come_back_in_insertion_order_across_writes(self):
        pytest.importorskip('numpy')
        dim = 6
        same = [1.0, 0.5, 0.25, 0.0, 0.0, 0.0]
        other = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        store = EmbeddingStore(db_path=':memory:', dimension=dim)
        for i in range(40):
            store.add(('t', i), 'x', vector=same if i % 2 == 0 else other)
        order = lambda **kw: [r.node_ref[1] for r in store.similarity_search(  # noqa: E731
            same, top_k=kw.pop('top_k', None), threshold=0.5, **kw)]
        assert order() == list(range(0, 40, 2))
        # update in place keeps the slot; delete + re-add moves to the end
        store.add(('t', 4), 'x', vector=same)
        store.remove(('t', 10))
        store.add(('t', 10), 'x', vector=same)
        store.add(('t', 99), 'x', vector=same)
        expected = [i for i in range(0, 40, 2) if i != 10] + [10, 99]
        assert order() == expected
        assert order(top_k=3) == expected[:3]
        assert order(top_k=len(expected)) == expected
        # an odd slot promoted to a tie joins in its own (original) position
        store.add(('t', 7), 'x', vector=same)
        expected.insert(expected.index(8), 7)
        assert order() == expected
        _assert_cache_matches_rebuild(store)
        store.close()

    def test_all_zero_vectors_and_zero_query(self):
        pytest.importorskip('numpy')
        store = EmbeddingStore(db_path=':memory:', dimension=3)
        store.add(('z', 1), 'zero', vector=[0.0, 0.0, 0.0])
        store.add(('z', 2), 'unit', vector=[1.0, 0.0, 0.0])
        assert [r.node_ref for r in store.similarity_search([1.0, 0.0, 0.0], top_k=None,
                                                            threshold=0.0)] \
            == [('z', 2), ('z', 1)]                 # the zero vector scores exactly 0.0
        assert [r.score for r in store.similarity_search([0.0, 0.0, 0.0], top_k=None,
                                                         threshold=0.0)] == [0.0, 0.0]
        assert store.similarity_search([0.0, 0.0, 0.0], threshold=0.1) == []
        store.close()

    def test_dimension_change_rebuilds_instead_of_patching(self):
        pytest.importorskip('numpy')
        store = EmbeddingStore(db_path=':memory:', dimension=4)
        for i in range(10):
            store.add(('d', i), 'x', vector=[1.0, i, 0.0, 0.0])
        assert len(store.similarity_search([1, 0, 0, 0], top_k=3)) == 3
        with pytest.raises(DimensionMismatchError):
            store.add(('d', 99), 'x', vector=[1.0, 2.0])
        with pytest.raises(DimensionMismatchError):
            store.similarity_search([1.0, 2.0])
        # the rejected write left the cache and the table untouched
        assert store.count() == 10
        _assert_cache_matches_rebuild(store)

        store.clear(reset_dimension=True)
        assert store.similarity_search([1, 0, 0, 0], top_k=3) == []
        model = _RefModel()
        rng = random.Random(5)
        for i in range(30):
            v = [rng.uniform(-1, 1) for _ in range(7)]
            store.add(('e', i), f'e{i}', vector=v)
            model.put(('e', i), v, f'e{i}')
            if i % 7 == 0:
                q = [rng.uniform(-1, 1) for _ in range(7)]
                _assert_same_ranking(store.similarity_search(q, top_k=5, threshold=-1.0),
                                     model.search(q, 5, None, -1.0))
        with pytest.raises(DimensionMismatchError):
            store.similarity_search([1, 0, 0, 0])
        _assert_cache_matches_rebuild(store)
        store.close()

    def test_growth_and_compaction_keep_the_cache_consistent(self):
        np = pytest.importorskip('numpy')
        rng = random.Random(11)
        dim = 5
        store = EmbeddingStore(db_path=':memory:', dimension=dim)
        model = _RefModel()
        for i in range(100):
            v = [rng.uniform(-1, 1) for _ in range(dim)]
            store.add(('g', i), f'g{i}', vector=v)
            model.put(('g', i), v, f'g{i}')
        q = [rng.uniform(-1, 1) for _ in range(dim)]
        store.similarity_search(q)                        # build with spare capacity
        capacity = store._cache.matrix.shape[0]
        # grow well past the initial capacity, one write at a time
        for i in range(100, capacity + 3000):
            v = [rng.uniform(-1, 1) for _ in range(dim)]
            store.add(('g', i), f'g{i}', vector=v)
            model.put(('g', i), v, f'g{i}')
        assert store._cache is not None and store._cache.matrix.shape[0] > capacity
        _assert_same_ranking(store.similarity_search(q, top_k=20, threshold=-1.0),
                             model.search(q, 20, None, -1.0))
        # delete a scattered half: tombstones, then compaction
        victims = [('g', i) for i in range(0, capacity + 3000, 2)]
        for ref in victims[:200]:
            store.remove(ref)
            model.drop(ref)
        assert store._cache.dead == 200
        _assert_same_ranking(store.similarity_search(q, top_k=20, threshold=-1.0),
                             model.search(q, 20, None, -1.0))
        for ref in victims[200:]:
            store.remove(ref)
            model.drop(ref)
        assert store._cache.dead < 40 or store._cache.dead * 4 <= store._cache.n
        _assert_same_ranking(store.similarity_search(q, top_k=None, threshold=-1.0),
                             model.search(q, None, None, -1.0))
        # delete everything, then start over on the same cache
        for ref in [r for r, *_ in list(model.rows.values())]:
            store.remove(ref)
            model.drop(ref)
        assert store.similarity_search(q, top_k=5, threshold=-1.0) == []
        assert store.score_map(q) == {}
        store.add(('again', 1), 'a', vector=[1.0] * dim)
        assert [r.node_ref for r in store.similarity_search(q, threshold=-1.0)] == [('again', 1)]
        _assert_cache_matches_rebuild(store)
        store.close()

    def test_writes_patch_the_cache_instead_of_dropping_it(self):
        pytest.importorskip('numpy')
        store = EmbeddingStore(db_path=':memory:', dimension=4)
        store.add_batch([(('p', i), 't', [1.0, i, 0.0, 1.0]) for i in range(50)])
        store.similarity_search([1, 0, 0, 0])
        cache = store._cache
        assert cache is not None
        store.add(('p', 50), 't', vector=[0.0, 1.0, 0.0, 0.0])
        store.add(('p', 3), 'new text', vector=[0.0, 1.0, 1.0, 0.0])
        store.remove(('p', 7))
        store.add_batch([(('q', 1), 'u', [1.0, 1.0, 1.0, 1.0])])
        assert store._cache is cache                     # same object, patched
        assert store.get(('p', 3)).text == 'new text'
        assert store.models() == {'provided': 51}
        top = store.similarity_search([0.0, 1.0, 1.0, 0.0], top_k=1, threshold=-1.0)[0]
        assert (top.node_ref, top.text) == (('p', 3), 'new text')

    def test_failed_write_does_not_leave_a_stale_cache(self):
        pytest.importorskip('numpy')
        store = EmbeddingStore(db_path=':memory:', dimension=3)
        store.add_batch([(('f', i), 't', [1.0, float(i), 0.0]) for i in range(10)])
        store.similarity_search([1, 0, 0])
        with pytest.raises(Exception):
            store.add_batch([(('f', 100), 't', [1.0, 2.0, 3.0]),
                             (('f', 101), 't', [1.0, 'x', 3.0])])
        refs_now = {r.node_ref for r in store.similarity_search(
            [1, 0, 0], top_k=None, threshold=-1.0)}
        # whatever the connection kept of the half-done batch, the cache must agree
        rows = store.export_embeddings()['embeddings']
        assert refs_now == {(r['type'], r['id']) for r in rows}
        _assert_cache_matches_rebuild(store)
        store.close()

    def test_models_counts_follow_writes_without_a_requery(self):
        pytest.importorskip('numpy')
        store = EmbeddingStore(db_path=':memory:', dimension=3, encoder=MockEncoder(3))
        store.add(('m', 1), 'a')
        store.add(('m', 2), 'b')
        assert store.models() == {store._model_name(): 2}        # primes the cache
        store.add(('m', 2), 'b again')                           # same model: no change
        store.remove(('m', 1))
        assert store.models() == {store._model_name(): 1}
        store.remove(('m', 2))
        assert store.models() == {}
        store.add(('m', 3), 'c', vector=[1.0, 2.0, 3.0])
        assert sum(store.models().values()) == 1

    def test_node_filter_search_matches_brute_force(self):
        pytest.importorskip('numpy')
        rng = random.Random(3)
        dim = 8
        graph = SemanticGraph(embedding_db=':memory:', encoder=MockEncoder(dim))
        model = _RefModel()
        for i in range(400):
            kind = 'keep' if i % 7 == 0 else 'drop'
            graph.add_node('item', i, kind=kind)
            v = [rng.uniform(-1, 1) for _ in range(dim)]
            graph.embedding_store.add(('item', i), f'i{i}', vector=v)
            model.put(('item', i), v, f'i{i}')
        graph.embedding_store.add(('ghost', 1), 'g', vector=[1.0] * dim)   # no node
        q = [rng.uniform(-1, 1) for _ in range(dim)]
        keep = lambda n: n.properties['kind'] == 'keep'          # noqa: E731
        everything = [e for e in model.search(q, None, None, -1.0)
                      if e[1][1] % 7 == 0]
        for top_k in (1, 5, 30, 57, 500, None):
            got = graph.similarity_search(q, top_k=top_k, threshold=-1.0, node_filter=keep)
            want = everything if top_k is None else everything[:top_k]
            _assert_same_ranking(got, want)
        assert graph.similarity_search(q, top_k=0, threshold=-1.0, node_filter=keep) == []
        # mutate, then filter again: a write between searches is visible
        graph.embedding_store.add(('item', 7), 'moved', vector=[v * 1.0 for v in q])
        top = graph.similarity_search(q, top_k=1, threshold=-1.0, node_filter=keep)[0]
        assert top.node_ref == ('item', 7) and top.text == 'moved'
        graph.close()

    def test_interleaved_writers_and_readers_leave_a_consistent_cache(self):
        np = pytest.importorskip('numpy')
        dim = 16
        store = EmbeddingStore(db_path=':memory:', dimension=dim)
        seed = random.Random(0)
        store.add_batch([(('s', i), 'x', [seed.uniform(-1, 1) for _ in range(dim)])
                         for i in range(300)])
        store.similarity_search([1.0] * dim)                     # cache exists
        errors = []

        def writer(tid):
            rng = random.Random(100 + tid)
            try:
                for i in range(200):
                    ref = ('w', tid * 1000 + rng.randrange(60))
                    r = rng.random()
                    if r < 0.5:
                        store.add(ref, f't{i}', vector=[rng.uniform(-1, 1) for _ in range(dim)])
                    elif r < 0.8:
                        store.remove(ref)
                    else:
                        store.add_batch([(ref, 'b', [rng.uniform(-1, 1) for _ in range(dim)])])
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        def reader(tid):
            rng = random.Random(500 + tid)
            try:
                for _ in range(150):
                    q = [rng.uniform(-1, 1) for _ in range(dim)]
                    res = store.similarity_search(q, top_k=7, threshold=-1.0)
                    scores = [r.score for r in res]
                    assert scores == sorted(scores, reverse=True)
                    assert len({r.node_ref for r in res}) == len(res)
                    store.score_map(q, node_type='w')
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(3)] + \
                  [threading.Thread(target=reader, args=(t,)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        cache = store._cache
        assert cache is not None and cache.n - cache.dead == store.count()
        _assert_cache_matches_rebuild(store)
        # and the final state agrees with an independent brute-force read of the table
        rows = store.export_embeddings()['embeddings']
        model = _RefModel()
        for row in rows:
            model.put((row['type'], row['id']), row['vector'], row['text'])
        q = [random.Random(9).uniform(-1, 1) for _ in range(dim)]
        _assert_same_ranking(store.similarity_search(q, top_k=25, threshold=-1.0),
                             model.search(q, 25, None, -1.0))
        store.close()

    def test_pure_python_fallback_still_agrees(self, monkeypatch):
        import isongraph_vector as iv
        monkeypatch.setattr(iv, '_np', None)
        rng = random.Random(8)
        dim = 6
        store = EmbeddingStore(db_path=':memory:', dimension=dim)
        model = _RefModel()
        for i in range(60):
            v = [rng.uniform(-1, 1) for _ in range(dim)]
            ref = ('a' if i % 2 else 'b', i)
            store.add(ref, f'x{i}', vector=v)
            model.put(ref, v, f'x{i}')
            if i % 9 == 0:
                q = [rng.uniform(-1, 1) for _ in range(dim)]
                _assert_same_ranking(store.similarity_search(q, top_k=7, threshold=-1.0),
                                     model.search(q, 7, None, -1.0))
                _assert_same_ranking(
                    store.similarity_search(q, top_k=None, node_type='a', threshold=0.0),
                    model.search(q, None, 'a', 0.0))
        store.remove(('a', 1))
        model.drop(('a', 1))
        q = [rng.uniform(-1, 1) for _ in range(dim)]
        _assert_same_ranking(store.similarity_search(q, top_k=5, threshold=-1.0),
                             model.search(q, 5, None, -1.0))
        assert len(store.score_map(q)) == len(model.rows)
        store.close()


# =============================================================================
# Approximate (IVF) index
# =============================================================================

#: Recall@10 the tests demand of the index on the clustered data below
#: (4,000 vectors, 40 clusters, 64 cells, nprobe 8 = 12.5% of the cells).
#: Measured: 1.000 on every seed tried; the floor leaves room for platform
#: differences in BLAS rounding, not for a worse algorithm.
ANN_RECALL_FLOOR = 0.95


def _clustered(n, dim=32, clusters=40, seed=1, noise=1.0):
    """Unit-ish clustered vectors plus a function that draws queries from them."""
    np = pytest.importorskip('numpy')
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def draw(count):
        which = rng.integers(0, clusters, count)
        pts = centers[which] + noise * rng.standard_normal((count, dim)).astype(np.float32) / np.sqrt(dim)
        return pts.astype(np.float32)

    return draw(n), draw


def _ann_store(n=4000, dim=32, seed=1, types=None, **kwargs):
    kwargs.setdefault('ann', 'ivf')
    kwargs.setdefault('ann_min_size', 0)
    kwargs.setdefault('ann_nlist', 64)
    kwargs.setdefault('ann_nprobe', 8)
    data, draw = _clustered(n, dim=dim, seed=seed)
    store = EmbeddingStore(db_path=':memory:', **kwargs)
    store.add_batch([((types(i) if types else 'doc', i), f'text {i}', data[i].tolist())
                     for i in range(n)])
    return store, data, draw


class TestApproximateIndex:
    """The opt-in IVF index: recall, exactness at nprobe == nlist, upkeep, filters."""

    @pytest.fixture(autouse=True)
    def _need_numpy(self):
        pytest.importorskip('numpy')

    def test_off_by_default_and_exact_is_untouched(self):
        store, data, draw = _ann_store(ann=None)
        assert store.ann is None
        assert store.ann_stats()['enabled'] is False
        with pytest.raises(Exception, match='ann'):
            store.build_ann_index()
        # Identical (not merely close) to the same store with the flag on and
        # exact=True: the exact path is the same code with or without ann.
        other, _, _ = _ann_store(ann='ivf')
        for q in draw(5).tolist():
            a = store.similarity_search(q, 10)
            b = other.similarity_search(q, 10, exact=True)
            assert [(r.node_ref, r.score) for r in a] == [(r.node_ref, r.score) for r in b]

    def test_argument_validation(self):
        with pytest.raises(ValueError):
            EmbeddingStore(ann='hnsw')
        with pytest.raises(ValueError):
            EmbeddingStore(ann_nprobe=0)
        with pytest.raises(ValueError):
            EmbeddingStore(ann='ivf', ann_nlist=0)
        with pytest.raises(ValueError):
            EmbeddingStore(ann='ivf', ann_min_size=-1)
        with pytest.raises(ValueError):
            EmbeddingStore(ann='ivf', sqlite_vec=True)
        store = EmbeddingStore(ann='ivf')
        with pytest.raises(ValueError):
            store.ann_nprobe = 0
        store.ann_nprobe = 5
        assert store.ann_nprobe == 5

    def test_ann_without_numpy_is_refused(self, monkeypatch):
        import isongraph_vector as iv
        monkeypatch.setattr(iv, '_np', None)
        with pytest.raises(iv.EmbeddingError, match='numpy'):
            EmbeddingStore(ann='ivf')
        assert EmbeddingStore().ann is None            # exact search needs no numpy

    def test_recall_floor_on_clustered_data(self):
        store, data, draw = _ann_store()
        report = store.recall_check(draw(60).tolist(), k=10)
        assert report.approximate is True
        assert report.nlist == 64 and report.nprobe == 8
        assert report.recall >= ANN_RECALL_FLOOR, report
        assert 0.0 <= report.min_recall <= report.recall <= 1.0
        stats = store.ann_stats()
        assert stats['active'] and stats['nlist'] == 64 and stats['trained_n'] == 4000
        assert stats['index_bytes'] > 0 and stats['searches']['approximate'] >= 60

    def test_results_carry_exact_scores(self):
        store, data, draw = _ann_store()
        for q in draw(10).tolist():
            exact = {r.node_ref: r.score for r in store.similarity_search(q, 50, exact=True)}
            for r in store.similarity_search(q, 10):
                if r.node_ref in exact:
                    assert r.score == pytest.approx(exact[r.node_ref], abs=1e-6)
            scores = [r.score for r in store.similarity_search(q, 10)]
            assert scores == sorted(scores, reverse=True)

    def test_probing_every_cell_is_the_exact_scan(self):
        import numpy as np
        store, data, draw = _ann_store(ann_nprobe=64)         # nprobe == nlist
        reference = EmbeddingStore(db_path=':memory:')
        reference.add_batch([(('doc', i), f'text {i}', data[i].tolist()) for i in range(len(data))])
        for q in draw(8).tolist():
            got = store.similarity_search(q, 10)
            want = reference.similarity_search(q, 10)
            assert [(r.node_ref, r.score, r.text) for r in got] == \
                   [(r.node_ref, r.score, r.text) for r in want]
        assert store.recall_check(draw(5).tolist()).approximate is False
        # and the candidate machinery itself, asked for every cell, sees every row
        cache = store._ensure_cache()
        ivf = store._ann_index(cache)
        rows, scores = store._ann_scores(cache, ivf, draw(1)[0], None, -1.0, ivf.nlist)
        assert rows.tolist() == list(range(len(data)))
        exact_scores = store._np_scores(cache, draw(1)[0])
        assert np.isfinite(scores).all() and exact_scores.shape == scores.shape

    def test_under_min_size_stays_exact(self):
        store, data, draw = _ann_store(n=500, ann_min_size=1000)
        q = draw(1)[0].tolist()
        assert [r.node_ref for r in store.similarity_search(q, 10)] == \
               [r.node_ref for r in store.similarity_search(q, 10, exact=True)]
        assert store.ann_stats()['active'] is False
        assert store.recall_check([q]).approximate is False

    def test_top_k_none_and_exact_flag_scan(self):
        store, data, draw = _ann_store()
        q = draw(1)[0].tolist()
        before = store.ann_stats()['searches']['approximate']
        assert len(store.similarity_search(q, None, threshold=-1.0)) == 4000
        store.similarity_search(q, 5, exact=True)
        assert store.ann_stats()['searches']['approximate'] == before
        assert store.similarity_search(q, 0) == []

    def test_zero_query_falls_back_to_exact(self):
        store, data, draw = _ann_store()
        res = store.similarity_search([0.0] * 32, 5, threshold=0.0)
        assert [r.score for r in res] == [0.0] * 5

    def test_writes_are_visible_immediately(self):
        store, data, draw = _ann_store()
        store.build_ann_index()
        fresh = draw(1)[0].tolist()
        store.add(('new', 1), 'fresh', vector=fresh)
        top = store.similarity_search(fresh, 3)
        assert top[0].node_ref == ('new', 1) and top[0].score == pytest.approx(1.0, abs=1e-5)
        assert store.ann_stats()['pending'] == 1

        # an in-place update moves the row: found at its new place, scored
        # from its new vector everywhere, and listed once
        target = draw(1)[0].tolist()
        store.add(('doc', 7), 'moved', vector=target)
        hit = store.similarity_search(target, 1)[0]
        assert hit.node_ref == ('doc', 7) and hit.score == pytest.approx(1.0, abs=1e-5)
        for q in [data[7].tolist(), target]:
            res = store.similarity_search(q, 400, threshold=-1.0)
            refs = [r.node_ref for r in res]
            assert len(refs) == len(set(refs))               # no stale second entry
            truth = {r.node_ref: r.score
                     for r in store.similarity_search(q, None, threshold=-1.0)}
            assert all(r.score == pytest.approx(truth[r.node_ref], abs=1e-6) for r in res)
            cache = store._ensure_cache()
            rows, _ = store._ann_scores(cache, cache.ivf, q, None, -1.0, 8)
            assert len(set(rows.tolist())) == rows.size

    def test_deletes_never_come_back(self):
        store, data, draw = _ann_store()
        store.build_ann_index()
        gone = set()
        for i in range(0, 4000, 3):                         # a third of them: forces compaction
            store.remove(('doc', i))
            gone.add(('doc', i))
        builds = store.ann_stats()['searches']['builds']
        for q in draw(15).tolist():
            got = {r.node_ref for r in store.similarity_search(q, 10)}
            assert not (got & gone)
            # a tombstone scores 0.0, which beats any negative cosine: asking
            # for many results with a negative threshold would surface one
            deep = {r.node_ref for r in store.similarity_search(q, 250, threshold=-1.0)}
            assert not (deep & gone)
            # the candidate rows themselves: only live ones, each once (the
            # lists still hold stale entries for rows deleted since the last regroup)
            cache = store._ensure_cache()
            rows, _ = store._ann_scores(cache, cache.ivf, q, None, -1.0, 8)
            assert cache.alive[rows].all() and len(set(rows.tolist())) == rows.size
        report = store.recall_check(draw(40).tolist())
        assert report.recall >= ANN_RECALL_FLOOR, report
        # the cache really compacted, and the index followed it
        cache = store._cache
        assert cache.dead == 0 or cache.dead * 4 <= cache.n
        assert cache.ivf is not None
        assert store.ann_stats()['searches']['builds'] == builds     # followed, not retrained
        assert (cache.ivf.assign[:cache.n][~cache.alive[:cache.n]] == -1).all()

    def test_pending_rows_fold_and_heavy_churn_retrains(self):
        store, data, draw = _ann_store(n=3000)
        store.build_ann_index()
        builds = store.ann_stats()['searches']['builds']
        extra = draw(700)
        store.add_batch([(('late', i), '', extra[i].tolist()) for i in range(700)])
        # folded at write time, 256 at a time: 700 = 2 x 256 + 188 still pending
        assert store.ann_stats()['pending'] == 188
        store.similarity_search(extra[0].tolist(), 5)        # a search folds nothing
        stats = store.ann_stats()
        assert stats['pending'] == 188 and stats['searches']['builds'] == builds
        more = draw(1200)
        store.add_batch([(('later', i), '', more[i].tolist()) for i in range(1200)])
        store.similarity_search(more[0].tolist(), 5)         # churn 1900 > 50% of 3000
        assert store.ann_stats()['searches']['builds'] == builds + 1
        assert store.ann_stats()['trained_n'] == 4900
        assert store.recall_check(draw(30).tolist()).recall >= ANN_RECALL_FLOOR

    def test_clear_and_dimension_change(self):
        store, data, draw = _ann_store(n=1500, ann_nlist=16, ann_nprobe=4)
        store.build_ann_index()
        store.clear()
        assert store.similarity_search(data[0].tolist(), 5) == []
        assert store.ann_stats()['active'] is False
        store.clear(reset_dimension=True)
        small, draw8 = _clustered(1200, dim=8, clusters=12, seed=3)
        store.add_batch([(('doc', i), '', small[i].tolist()) for i in range(1200)])
        res = store.similarity_search(small[5].tolist(), 5)
        assert res[0].node_ref == ('doc', 5)
        assert store.ann_stats()['active'] and store.dimension == 8

    def test_node_type_filter_and_widening(self):
        store, data, draw = _ann_store(types=lambda i: 'rare' if i % 200 == 0 else 'common',
                                       ann_nprobe=2)
        q = draw(1)[0].tolist()
        common = store.similarity_search(q, 10, node_type='common')
        assert len(common) == 10 and {r.node_ref[0] for r in common} == {'common'}
        # 20 rare rows in the whole store, a top_k of 15: the two probed cells
        # cannot hold 15 of them, so the probe must widen (ending in the scan)
        widened_before = store.ann_stats()['searches']['widened']
        rare = store.similarity_search(q, 15, node_type='rare', threshold=-1.0)
        assert len(rare) == 15 and {r.node_ref[0] for r in rare} == {'rare'}
        exact = store.similarity_search(q, 15, node_type='rare', threshold=-1.0, exact=True)
        assert len(rare) == len(exact)
        counts = store.ann_stats()['searches']
        assert counts['widened'] > widened_before or counts['exact'] > 0
        assert store.similarity_search(q, 5, node_type='nope') == []

    def test_threshold_is_respected(self):
        store, data, draw = _ann_store()
        q = draw(1)[0].tolist()
        best = store.similarity_search(q, 1, exact=True)[0].score
        res = store.similarity_search(q, 10, threshold=best - 1e-3)
        assert all(r.score >= best - 1e-3 for r in res)
        assert store.similarity_search(q, 10, threshold=2.0) == []
        # fewer than k above a threshold: same short list as the scan
        cut = store.similarity_search(q, 10, exact=True)[4].score
        got = store.similarity_search(q, 10, threshold=cut)
        want = store.similarity_search(q, 10, threshold=cut, exact=True)
        assert {r.node_ref for r in got} == {r.node_ref for r in want}

    def test_recall_check_shapes(self):
        np = pytest.importorskip('numpy')
        store, data, draw = _ann_store()
        q = draw(3)
        assert store.recall_check(q[0].tolist()).queries == 1            # one flat vector
        assert store.recall_check(q[0]).queries == 1                      # one ndarray
        assert store.recall_check(q).queries == 3                         # a 2-D array
        off = EmbeddingStore(db_path=':memory:')
        off.add_batch([(('doc', i), '', data[i].tolist()) for i in range(100)])
        r = off.recall_check(q.tolist())
        assert r.recall == 1.0 and r.approximate is False
        with pytest.raises(ValueError):
            store.recall_check(q, k=0)

    def test_thread_safety_smoke(self):
        store, data, draw = _ann_store(n=3000, ann_nlist=32, ann_nprobe=4)
        errors = []
        extra = draw(600)
        queries = draw(40).tolist()

        def reader(seed):
            try:
                for i in range(120):
                    res = store.similarity_search(queries[(i + seed) % 40], 8, threshold=-1.0)
                    assert len(res) == 8
                    scores = [r.score for r in res]
                    assert scores == sorted(scores, reverse=True)
                    assert len({r.node_ref for r in res}) == len(res)
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        def writer(seed):
            try:
                for i in range(100):
                    store.add(('w', seed * 1000 + i), '', vector=extra[(seed * 100 + i) % 600].tolist())
                    if i % 4 == 0:
                        store.remove(('doc', seed * 300 + i))
                    if i % 20 == 0:
                        store.build_ann_index()
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=reader, args=(t,)) for t in range(3)] + \
                  [threading.Thread(target=writer, args=(t,)) for t in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        cache = store._cache
        assert cache.n - cache.dead == store.count()
        assert store.recall_check(queries[:20]).recall >= ANN_RECALL_FLOOR


class TestApproximateIndexInGraph:
    """The index through SemanticGraph: node_filter, passthrough, load."""

    @pytest.fixture(autouse=True)
    def _need_numpy(self):
        pytest.importorskip('numpy')

    @staticmethod
    def _graph(n=3000, dim=32, **kwargs):
        data, draw = _clustered(n, dim=dim)
        graph = SemanticGraph(name='ann', encoder=MockEncoder(dimension=dim), auto_embed=False,
                              ann='ivf', ann_min_size=0, ann_nlist=48, ann_nprobe=6, **kwargs)
        for i in range(n):
            graph.add_node('item', i, group=i % 50)
        graph.embedding_store.add_batch(
            [(('item', i), f'item {i}', data[i].tolist()) for i in range(n)])
        return graph, data, draw

    def test_constructor_passthrough(self):
        graph, data, draw = self._graph()
        store = graph.embedding_store
        assert store.ann == 'ivf' and store.ann_nprobe == 6
        res = graph.similarity_search(draw(1)[0].tolist(), 5)
        assert len(res) == 5 and store.ann_stats()['active']
        assert graph.recall_check(draw(20).tolist()).recall >= ANN_RECALL_FLOOR
        assert SemanticGraph().embedding_store.ann is None

    def test_exact_flag_and_similar_to_node(self):
        graph, data, draw = self._graph()
        q = draw(1)[0].tolist()
        before = graph.embedding_store.ann_stats()['searches']['approximate']
        graph.similarity_search(q, 5, exact=True)
        graph.similarity_search(q, 5, exact=True, node_filter=lambda n: True)
        assert graph.embedding_store.ann_stats()['searches']['approximate'] == before
        near = graph.similar_to_node(('item', 3), top_k=5)
        assert len(near) == 5 and ('item', 3) not in {r.node_ref for r in near}

    def test_node_filter_gets_k_results_even_when_selective(self):
        graph, data, draw = self._graph()
        q = draw(1)[0].tolist()
        # 1 node in 50 passes: the probed cells hold few of them, so the
        # search has to widen until 10 pass
        res = graph.similarity_search(q, 10, node_filter=lambda n: n.properties['group'] == 7)
        assert len(res) == 10
        assert all(graph.get_node_by_ref(r.node_ref).properties['group'] == 7 for r in res)
        want = graph.similarity_search(q, 10, exact=True,
                                       node_filter=lambda n: n.properties['group'] == 7)
        assert len({r.node_ref for r in res} & {r.node_ref for r in want}) >= 7
        scores = [r.score for r in res]
        assert scores == sorted(scores, reverse=True)
        # a filter nothing passes: terminates, empty
        assert graph.similarity_search(q, 10, node_filter=lambda n: False) == []
        # ...and a lenient one agrees closely with the scan
        got = graph.similarity_search(q, 10, node_filter=lambda n: n.properties['group'] % 2 == 0)
        ref = graph.similarity_search(q, 10, exact=True,
                                      node_filter=lambda n: n.properties['group'] % 2 == 0)
        assert len({r.node_ref for r in got} & {r.node_ref for r in ref}) >= 8

    def test_semantic_multi_hop_uses_index_for_seeds(self):
        graph, data, draw = self._graph(n=2000)
        for i in range(1999):
            graph.add_edge('NEXT', ('item', i), ('item', i + 1))
        q = draw(1)[0].tolist()
        out = graph.semantic_multi_hop(q, rel_type='NEXT', max_hops=2, top_k_seeds=3,
                                       threshold=-1.0)
        assert out and out[0].hop_count == 0
        assert graph.embedding_store.ann_stats()['searches']['approximate'] >= 1
        # blend > 0 scores every node and must not touch the approximate path
        before = graph.embedding_store.ann_stats()['searches']['approximate']
        graph.semantic_multi_hop(q, rel_type='NEXT', max_hops=1, blend=0.3, threshold=-1.0)
        assert graph.embedding_store.ann_stats()['searches']['approximate'] == before

    def test_load_passes_ann_and_rebuilds_lazily(self, tmp_path):
        graph, data, draw = self._graph(n=1500)
        path = tmp_path / 'g.isong'
        graph.save(path)
        loaded = SemanticGraph.load(path, encoder=graph.encoder, ann='ivf', ann_min_size=0,
                                    ann_nlist=32, ann_nprobe=4)
        assert loaded.embedding_store.ann == 'ivf'
        assert loaded.embedding_store.ann_stats()['active'] is False      # nothing persisted
        assert len(loaded.similarity_search(draw(1)[0].tolist(), 5)) == 5
        assert loaded.embedding_store.ann_stats()['active'] is True


# =============================================================================
# HttpEncoder (against a local http.server; the tests never leave the machine)
# =============================================================================

class _FakeEmbeddingServer:
    """A throwaway HTTP server on 127.0.0.1 that plays back programmed replies."""

    def __init__(self):
        import http.server
        import json as _json
        import threading as _threading
        self.requests = []          # (path, headers dict, parsed body)
        self.script = []            # callables(request_index, body) -> (status, headers, bytes)
        self.default = None
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(length)
                try:
                    body = _json.loads(raw)
                except ValueError:
                    body = None
                index = len(outer.requests)
                outer.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                step = outer.script[index] if index < len(outer.script) else outer.default
                status, headers, payload = step(index, body)
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = _threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _vec_for(text, dim=4):
    """A stable little vector per text, so tests can tell which text it came from."""
    base = float(sum(map(ord, text)) % 97) + 1.0
    return [base + j for j in range(dim)]


def _openai_reply(dim=4, shuffle=False, with_index=True):
    import json as _json

    def step(index, body):
        items = [{'object': 'embedding', 'embedding': _vec_for(t, dim), **({'index': i} if with_index else {})}
                 for i, t in enumerate(body['input'])]
        if shuffle:
            items.reverse()
        return 200, {'Content-Type': 'application/json'}, _json.dumps(
            {'object': 'list', 'data': items, 'model': body['model']}).encode()
    return step


def _ollama_reply(dim=4):
    import json as _json

    def step(index, body):
        return 200, {'Content-Type': 'application/json'}, _json.dumps(
            {'model': body['model'], 'embeddings': [_vec_for(t, dim) for t in body['input']]}).encode()
    return step


def _json_reply(obj, status=200, headers=None):
    import json as _json
    return lambda index, body: (status, {'Content-Type': 'application/json', **(headers or {})},
                                 _json.dumps(obj).encode())


@pytest.fixture
def fake_server():
    server = _FakeEmbeddingServer()
    yield server
    server.close()


class TestHttpEncoder:
    def _encoder(self, server, path='/v1/embeddings', **kwargs):
        from isongraph_vector import HttpEncoder
        kwargs.setdefault('model', 'test-embed')
        return HttpEncoder(server.base + path, **kwargs)

    def test_openai_request_and_response(self, fake_server):
        fake_server.default = _openai_reply()
        enc = self._encoder(fake_server, api_key='sk-secret', dimension=4, version='2025-01')
        assert enc.model_name == 'test-embed' and enc.version == '2025-01' and enc.dimension == 4
        vec = enc.encode('hello')
        assert vec == _vec_for('hello') and all(isinstance(x, float) for x in vec)
        path, headers, body = fake_server.requests[0]
        assert path == '/v1/embeddings'
        assert body == {'model': 'test-embed', 'input': ['hello']}
        assert headers['authorization'] == 'Bearer sk-secret'
        assert headers['content-type'] == 'application/json'

    def test_openai_data_is_sorted_by_index(self, fake_server):
        fake_server.default = _openai_reply(shuffle=True)
        enc = self._encoder(fake_server)
        texts = ['alpha', 'bravo', 'charlie']
        assert enc.encode_batch(texts) == [_vec_for(t) for t in texts]

    def test_openai_without_index_keeps_order(self, fake_server):
        fake_server.default = _openai_reply(with_index=False)
        enc = self._encoder(fake_server)
        assert enc.encode_batch(['a', 'bb']) == [_vec_for('a'), _vec_for('bb')]

    def test_ollama_format(self, fake_server):
        fake_server.default = _ollama_reply()
        enc = self._encoder(fake_server, path='/api/embed', api='ollama')
        assert enc.encode_batch(['x', 'yy']) == [_vec_for('x'), _vec_for('yy')]
        assert 'authorization' not in fake_server.requests[0][1]
        assert fake_server.requests[0][2] == {'model': 'test-embed', 'input': ['x', 'yy']}

    def test_batches_are_split_and_order_kept(self, fake_server):
        fake_server.default = _openai_reply()
        enc = self._encoder(fake_server, batch_size=2)
        texts = [f't{i}' for i in range(5)]
        assert enc.encode_batch(texts) == [_vec_for(t) for t in texts]
        assert [len(r[2]['input']) for r in fake_server.requests] == [2, 2, 1]
        assert enc.encode_batch([]) == [] and len(fake_server.requests) == 3

    def test_dimension_is_learned_or_probed(self, fake_server):
        fake_server.default = _openai_reply(dim=6)
        enc = self._encoder(fake_server)
        assert enc.dimension == 6                    # one probe request
        assert len(fake_server.requests) == 1
        assert enc.encode('abc') and enc.dimension == 6 and len(fake_server.requests) == 2

    def test_dimension_mismatch_is_an_error(self, fake_server):
        from isongraph_vector import DimensionMismatchError
        fake_server.default = _openai_reply(dim=4)
        enc = self._encoder(fake_server, dimension=8)
        with pytest.raises(DimensionMismatchError, match='4 dimensions, expected 8'):
            enc.encode('x')
        # a reply whose own vectors disagree
        fake_server.requests.clear()
        fake_server.default = _json_reply({'data': [
            {'index': 0, 'embedding': [1.0, 2.0]}, {'index': 1, 'embedding': [1.0, 2.0, 3.0]}]})
        with pytest.raises(DimensionMismatchError):
            self._encoder(fake_server).encode_batch(['a', 'b'])

    def test_http_error_carries_status_and_never_the_key(self, fake_server):
        from isongraph_vector import HttpEncoderError
        fake_server.default = _json_reply(
            {'error': {'message': 'Incorrect API key provided: sk-secret-123'}}, status=401)
        enc = self._encoder(fake_server, api_key='sk-secret-123')
        with pytest.raises(HttpEncoderError) as info:
            enc.encode('x')
        assert info.value.status == 401
        assert 'HTTP 401' in str(info.value) and 'Incorrect API key' in str(info.value)
        assert 'sk-secret-123' not in str(info.value) and 'sk-secret-123' not in repr(enc)
        assert len(fake_server.requests) == 1                  # a 401 is not retried

    def test_error_message_hides_url_credentials_and_query(self, fake_server):
        from isongraph_vector import HttpEncoder, HttpEncoderError
        fake_server.default = _json_reply({'error': 'nope'}, status=403)
        host = fake_server.base.split('//')[1]
        enc = HttpEncoder(f'http://user:pw@{host}/v1/embeddings?key=URLSECRET', 'm')
        with pytest.raises(HttpEncoderError) as info:
            enc.encode('x')
        text = str(info.value) + repr(enc)
        assert 'URLSECRET' not in text and 'pw@' not in text and 'user:' not in text

    def test_retries_429_and_5xx_then_succeeds(self, fake_server, monkeypatch):
        from isongraph_vector import HttpEncoder
        monkeypatch.setattr(HttpEncoder, 'RETRY_BACKOFF', 0.0)
        fake_server.script = [_json_reply({'e': 1}, status=429), _json_reply({'e': 1}, status=503)]
        fake_server.default = _openai_reply()
        enc = self._encoder(fake_server)
        assert enc.encode('retry me') == _vec_for('retry me')
        assert len(fake_server.requests) == 3

    def test_retries_are_bounded(self, fake_server, monkeypatch):
        from isongraph_vector import HttpEncoder, HttpEncoderError
        monkeypatch.setattr(HttpEncoder, 'RETRY_BACKOFF', 0.0)
        fake_server.default = _json_reply({'e': 1}, status=500)
        with pytest.raises(HttpEncoderError) as info:
            self._encoder(fake_server, retries=2).encode('x')
        assert info.value.status == 500 and len(fake_server.requests) == 3
        fake_server.requests.clear()
        with pytest.raises(HttpEncoderError):
            self._encoder(fake_server, retries=0).encode('x')
        assert len(fake_server.requests) == 1
        fake_server.requests.clear()
        fake_server.default = _json_reply({'e': 1}, status=400)       # client errors: no retry
        with pytest.raises(HttpEncoderError):
            self._encoder(fake_server, retries=5).encode('x')
        assert len(fake_server.requests) == 1

    @pytest.mark.parametrize('reply', [
        lambda i, b: (200, {}, b'this is not json'),
        _json_reply([1, 2, 3]),
        _json_reply({'object': 'list'}),                                   # no data
        _json_reply({'data': [{'index': 0}]}),                             # no embedding
        _json_reply({'data': [{'index': 0, 'embedding': [1.0]}]}),         # wrong count for 2 texts
        _json_reply({'data': [{'index': 0, 'embedding': [1.0]}, {'index': 0, 'embedding': [1.0]}]}),
        _json_reply({'data': [{'index': 0, 'embedding': 'abc'}, {'index': 1, 'embedding': [1.0]}]}),
        _json_reply({'data': [{'index': 0, 'embedding': []}, {'index': 1, 'embedding': [1.0]}]}),
        _json_reply({'data': [{'index': 0, 'embedding': [True]}, {'index': 1, 'embedding': [1.0]}]}),
    ])
    def test_malformed_replies_raise(self, fake_server, reply):
        from isongraph_vector import HttpEncoderError
        fake_server.default = reply
        with pytest.raises(HttpEncoderError):
            self._encoder(fake_server).encode_batch(['a', 'b'])

    def test_nan_in_reply_is_rejected(self, fake_server):
        from isongraph_vector import HttpEncoderError
        fake_server.default = lambda i, b: (200, {}, b'{"data": [{"index": 0, "embedding": [1.0, NaN]}]}')
        with pytest.raises(HttpEncoderError, match='finite'):
            self._encoder(fake_server).encode('a')

    def test_wrong_api_for_the_reply(self, fake_server):
        from isongraph_vector import HttpEncoderError
        fake_server.default = _ollama_reply()
        with pytest.raises(HttpEncoderError, match="'data'"):
            self._encoder(fake_server).encode('a')                # api="openai" against an Ollama reply
        fake_server.default = _openai_reply()
        with pytest.raises(HttpEncoderError, match="'embeddings'"):
            self._encoder(fake_server, api='ollama').encode('a')

    def test_redirects_are_not_followed(self, fake_server):
        from isongraph_vector import HttpEncoderError
        fake_server.script = [lambda i, b: (302, {'Location': fake_server.base + '/elsewhere'}, b'')]
        fake_server.default = _openai_reply()
        enc = self._encoder(fake_server, api_key='sk-leak-me')
        with pytest.raises(HttpEncoderError, match='redirect') as info:
            enc.encode('private text')
        assert info.value.status == 302
        assert [r[0] for r in fake_server.requests] == ['/v1/embeddings']     # never reached /elsewhere

    def test_unreachable_endpoint_and_timeout(self, fake_server):
        from isongraph_vector import HttpEncoder, HttpEncoderError
        dead = _FakeEmbeddingServer()
        base = dead.base
        dead.close()
        with pytest.raises(HttpEncoderError, match='could not reach') as info:
            HttpEncoder(base + '/v1/embeddings', 'm', api_key='sk-xyz', timeout=2).encode('x')
        assert info.value.status is None and 'sk-xyz' not in str(info.value)

        import time as _time
        fake_server.default = lambda i, b: (_time.sleep(1.0), (200, {}, b'{}'))[1]
        with pytest.raises(HttpEncoderError, match='could not reach'):
            self._encoder(fake_server, timeout=0.2).encode('slow')

    def test_custom_headers_win(self, fake_server):
        fake_server.default = _openai_reply()
        enc = self._encoder(fake_server, api_key='ignored', headers={'Authorization': 'Token abc', 'X-Org': 'o1'})
        enc.encode('x')
        headers = fake_server.requests[0][1]
        assert headers['authorization'] == 'Token abc' and headers['x-org'] == 'o1'

    @pytest.mark.parametrize('kwargs', [
        {'url': 'ftp://host/x'}, {'url': 'file:///etc/passwd'}, {'url': 'not a url'},
        {'api': 'cohere'}, {'batch_size': 0}, {'timeout': 0}, {'dimension': 0}, {'retries': -1},
        {'model': ''},
    ])
    def test_constructor_validation(self, kwargs):
        from isongraph_vector import HttpEncoder
        args = {'url': 'http://localhost:1/x', 'model': 'm'}
        args.update(kwargs)
        with pytest.raises(ValueError):
            HttpEncoder(**args)

    def test_non_string_text_is_rejected(self, fake_server):
        fake_server.default = _openai_reply()
        with pytest.raises(TypeError):
            self._encoder(fake_server).encode_batch(['ok', 3])
        assert fake_server.requests == []

    def test_works_inside_a_semantic_graph(self, fake_server):
        fake_server.default = _openai_reply(dim=8)
        enc = self._encoder(fake_server, dimension=8)
        graph = SemanticGraph(encoder=enc, embed_fields=['name'])
        graph.add_node('person', 1, name='Alice')
        graph.add_node('person', 2, name='Bob')
        assert graph.embedding_store.models() == {'test-embed': 2}
        best = graph.similarity_search('Alice', top_k=1)[0]
        assert best.node_ref == ('person', 1) and best.score == pytest.approx(1.0, abs=1e-6)
        assert graph.embedding_store.stale() == []
        embeddings = graph.embedding_store.export_embeddings()
        assert embeddings['encoder_model'] == 'test-embed' and embeddings['dimension'] == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
