//! The memory layout: every vector is stored once. A counting allocator
//! measures the live heap, so the test sees the real cost of a store (and would
//! see a second copy of the vectors, which an `EmbeddingStore` method could not
//! report on its own behalf).
//!
//! This file holds a single test so no other test thread allocates meanwhile.

use isongraph_vector_rs::{AnnConfig, EmbeddingStore, NodeId};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size >= layout.size() {
            LIVE.fetch_add(new_size - layout.size(), Ordering::Relaxed);
        } else {
            LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn vector(i: usize, dim: usize) -> Vec<f32> {
    let mut x = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

#[test]
fn the_vectors_are_held_once() {
    const N: usize = 20_000;
    const DIM: usize = 384;
    let raw = N * DIM * 4;

    let before = LIVE.load(Ordering::Relaxed);
    let mut store = EmbeddingStore::new(None);
    for i in 0..N {
        store
            .add(NodeId::new("doc", i.to_string()), "", Some(vector(i, DIM)))
            .unwrap();
    }
    // the first search used to build a second, contiguous copy
    let q = vector(N + 1, DIM);
    store.similarity_search_vector(&q, 10, None, -1.0).unwrap();
    store
        .set_ann(Some(AnnConfig {
            nlist: Some(64),
            min_size: 0,
            ..AnnConfig::default()
        }))
        .unwrap();
    store.build_ann_index(false).unwrap();
    store.similarity_search_vector(&q, 10, None, -1.0).unwrap();
    let held = LIVE.load(Ordering::Relaxed) - before;
    let ratio = held as f64 / raw as f64;
    eprintln!("{N} x {DIM}: raw {raw} bytes, store holds {held} ({ratio:.3}x)");

    // one copy of the floats plus per-record metadata and the index: under 1.2x
    // (the old layout, records plus a scan cache, was about 2.1x)
    assert!(
        ratio < 1.2,
        "the store holds {ratio:.2}x the raw vector bytes"
    );

    // a record is a row of the matrix: the record built on request carries the vector
    let id = NodeId::new("doc", "123");
    assert_eq!(store.get(&id).unwrap().vector, vector(123, DIM));

    // an update overwrites its row in place: the heap does not grow
    let steady = LIVE.load(Ordering::Relaxed);
    for i in 0..2_000 {
        store
            .add(
                NodeId::new("doc", i.to_string()),
                "",
                Some(vector(i + 7, DIM)),
            )
            .unwrap();
    }
    let grown = LIVE.load(Ordering::Relaxed).abs_diff(steady);
    assert!(
        grown < 2_000 * 400,
        "2,000 in-place updates grew the heap by {grown} bytes"
    );
}
