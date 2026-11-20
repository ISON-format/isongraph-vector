//! Dump the HNSW answers for deterministic data, so another port can check that
//! it builds the same graph and returns the same top-k.
//!
//! ```text
//! cargo run --release --example hnsw_xcheck -- [n] [dim] [queries] [k] [m] [ef_construction] [seed] [ef_search, 0 = auto] [noise]
//! ```
//!
//! The data uses integers only (every component is `integer / 1024`, exact in
//! `f32`), drawn from the SplitMix64 generator of the level draw, so every port
//! can regenerate the same vectors with no floating-point library in between:
//! `clusters` centres of `dim` components in `[-1000, 1000]`, a row is a centre
//! plus noise in `[-noise, noise]` (500 by default). Queries come from the same clusters.
//!
//! Output, one line per query and phase: `<phase> q<i> id:score_bits ...` for
//! the HNSW answer and `<phase> x<i> ...` for the exact scan (the scores' bit
//! patterns show whether two ports' scorers agree to the last bit). Phases:
//! `built` (after `build_ann_index`), `writes` (inserts, updates, deletes
//! applied one by one, linked by the write path), `compacted` (30% more deleted,
//! which compacts the table and the graph).

use isongraph_vector_rs::{AnnConfig, EmbeddingStore, HnswConfig, NodeId};

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

const CLUSTERS: usize = 30;

fn centers(rng: &mut Rng, dim: usize) -> Vec<Vec<i32>> {
    (0..CLUSTERS)
        .map(|_| (0..dim).map(|_| rng.below(2001) as i32 - 1000).collect())
        .collect()
}

fn draw(rng: &mut Rng, centers: &[Vec<i32>], noise: usize) -> Vec<f32> {
    let c = &centers[rng.below(centers.len())];
    c.iter()
        .map(|&x| (x + rng.below(2 * noise + 1) as i32 - noise as i32) as f32 / 1024.0)
        .collect()
}

fn line(phase: &str, tag: char, i: usize, hits: &[isongraph_vector_rs::SimilarityResult]) {
    let mut out = format!("{phase} {tag}{i}");
    for h in hits {
        out.push_str(&format!(" {}:{:08x}", h.node_id.id, h.score.to_bits()));
    }
    println!("{out}");
}

fn dump(phase: &str, store: &EmbeddingStore, queries: &[Vec<f32>], k: usize) {
    for (i, q) in queries.iter().enumerate() {
        let hits = store.similarity_search_vector(q, k, None, -2.0).unwrap();
        line(phase, 'q', i, &hits);
    }
    for (i, q) in queries.iter().enumerate() {
        let hits = store
            .similarity_search_vector_exact(q, k, None, -2.0)
            .unwrap();
        line(phase, 'x', i, &hits);
    }
    let s = store.ann_stats();
    println!(
        "{phase} stats nodes={} dead={} level={:?} approximate={}",
        s.graph_nodes, s.graph_dead, s.max_level, s.searches.approximate
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get =
        |i: usize, d: usize| -> usize { args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d) };
    let (n, dim, nq, k, m, efc, seed, ef, noise) = (
        get(1, 5000),
        get(2, 32),
        get(3, 100),
        get(4, 10),
        get(5, 16),
        get(6, 200),
        get(7, 1) as u64,
        get(8, 0),
        get(9, 500),
    );
    let cs = centers(&mut Rng(seed ^ 0x51), dim);
    let mut data = Rng(seed ^ 0x52);
    let mut qrng = Rng(seed ^ 0x53);
    let queries: Vec<Vec<f32>> = (0..nq).map(|_| draw(&mut qrng, &cs, noise)).collect();

    let mut store = EmbeddingStore::new(None);
    store
        .set_ann(Some(AnnConfig {
            min_size: 0,
            hnsw: HnswConfig {
                m,
                ef_construction: efc,
                ef_search: if ef == 0 { None } else { Some(ef) },
            },
            ..AnnConfig::hnsw()
        }))
        .unwrap();
    for i in 0..n {
        store
            .add(
                NodeId::new("doc", i.to_string()),
                "",
                Some(draw(&mut data, &cs, noise)),
            )
            .unwrap();
    }
    store.build_ann_index(false).unwrap();
    println!("params n={n} dim={dim} queries={nq} k={k} m={m} efc={efc} seed={seed} ef={ef} noise={noise}");
    dump("built", &store, &queries, k);

    // writes: inserts, updates, deletes, one at a time
    let mut ops = Rng(seed ^ 0x54);
    for i in 0..n / 10 {
        store
            .add(
                NodeId::new("doc", (n + i).to_string()),
                "",
                Some(draw(&mut ops, &cs, noise)),
            )
            .unwrap();
    }
    for i in 0..n / 20 {
        store
            .add(
                NodeId::new("doc", ((i * 13) % n).to_string()),
                "",
                Some(draw(&mut ops, &cs, noise)),
            )
            .unwrap();
    }
    for i in 0..n / 25 {
        store.remove(&NodeId::new("doc", ((i * 17 + 5) % n).to_string()));
    }
    dump("writes", &store, &queries, k);

    // enough deletes to compact (more than a quarter of the slots dead)
    for i in 0..n {
        if i % 3 == 1 {
            store.remove(&NodeId::new("doc", i.to_string()));
        }
    }
    store.verify_invariants().unwrap();
    dump("compacted", &store, &queries, k);
}
