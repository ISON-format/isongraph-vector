//! Write the golden snapshot fixture the other ports must read.
//!
//! Usage: cargo run --example snapshot_fixture -- <output-dir>
//!
//! Produces `golden.ison`, `golden.ison.embeddings.json` and
//! `golden.ison.embeddings.snap`: a small graph whose store holds ids with
//! colons and unicode, empty / quoted / multi-line / unicode text, an exact
//! duplicate vector (a tie), a zero vector, two encoder models, updates and
//! tombstones. Every port's test loads the three files, expects the snapshot
//! to be used, and expects the same store as a JSON-only load.

use isongraph_vector_rs::{MockEncoder, NodeId, SemanticGraph};

const DIM: usize = 12;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: snapshot_fixture <output-dir>")?;
    std::fs::create_dir_all(&dir)?;
    let n = 40usize;
    let mut g = SemanticGraph::new("golden", Box::new(MockEncoder::new(DIM)));
    g.add_node_with_embed("person", "1", vec![("name", "Alice")], None)?;
    g.add_node_with_embed("person", "2", vec![("name", "Bob")], None)?;
    g.add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])?;
    let mut rng = Rng(7);
    let types = ["doc", "note", "a:b"];
    let id_of = |i: usize| {
        if matches!(i % 9, 0) {
            format!("id:with:colons:{i}")
        } else {
            format!("{i:05}")
        }
    };
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for i in 0..n {
        let v: Vec<f32> = if i % 17 == 5 && !vectors.is_empty() {
            vectors[i / 2].clone()
        } else if i % 41 == 7 || i == 7 {
            vec![0.0; DIM]
        } else {
            (0..DIM).map(|_| rng.f()).collect()
        };
        vectors.push(v.clone());
        if i == n / 2 {
            g.set_encoder(Box::new(
                MockEncoder::new(DIM).with_model_name("second-model"),
            ));
        }
        let text = match i % 5 {
            0 => String::new(),
            1 => format!("quote \" and \\ and\nnewline {i}"),
            2 => format!("unicode \u{1F600} \u{00E9}\u{4E2D} {i}"),
            _ => format!("text {i}"),
        };
        g.embedding_store_mut()
            .add(NodeId::new(types[i % 3], id_of(i)), &text, Some(v))?;
    }
    for i in (0..n).step_by(11) {
        let v: Vec<f32> = (0..DIM).map(|_| rng.f()).collect();
        g.embedding_store_mut()
            .add(NodeId::new(types[i % 3], id_of(i)), "updated", Some(v))?;
    }
    for i in (3..n).step_by(7) {
        g.embedding_store_mut()
            .remove(&NodeId::new(types[i % 3], id_of(i)));
    }
    let path = std::path::Path::new(&dir).join("golden.ison");
    g.save_with_snapshot(&path, None)?;
    println!(
        "wrote {} (+ .embeddings.json, .embeddings.snap)",
        path.display()
    );
    Ok(())
}
