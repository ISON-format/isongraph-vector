//! `HttpEncoder` against a local mock server: `cargo test --features http-encoder`.
//!
//! No real network. Each test starts a `TcpListener` on 127.0.0.1 that plays a
//! script of canned responses and records what it was sent.

#![cfg(feature = "http-encoder")]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use isongraph_vector_rs::{
    EmbeddingEncoder, EmbeddingError, EmbeddingStore, HttpEncoder, HttpEncoderError, NodeId,
    SemanticGraph,
};

#[derive(Clone)]
struct Canned {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: String,
}

fn ok(body: &str) -> Canned {
    Canned {
        status: 200,
        headers: vec![],
        body: body.to_string(),
    }
}

fn status(code: u16, body: &str) -> Canned {
    Canned {
        status: code,
        headers: vec![],
        body: body.to_string(),
    }
}

#[derive(Clone, Debug)]
struct Recorded {
    request_line: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl Recorded {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).unwrap()
    }
}

struct Mock {
    url: String,
    seen: Arc<Mutex<Vec<Recorded>>>,
    handle: Option<JoinHandle<()>>,
}

impl Mock {
    /// Answers the first connection with `script[0]`, the next with `script[1]`
    /// and so on, then stops listening.
    fn start(script: Vec<Canned>) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/embeddings", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let handle = thread::spawn(move || {
            for canned in script {
                let (mut stream, _) = listener.accept().unwrap();
                let recorded = read_request(&mut stream);
                log.lock().unwrap().push(recorded);
                write_response(&mut stream, &canned);
            }
        });
        Mock {
            url,
            seen,
            handle: Some(handle),
        }
    }

    fn requests(&self) -> Vec<Recorded> {
        if let Some(handle) = self.handle.as_ref() {
            // The script is finished once the thread is; give it a moment.
            let deadline = Instant::now() + Duration::from_secs(5);
            while !handle.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
        }
        self.seen.lock().unwrap().clone()
    }
}

fn read_request(stream: &mut TcpStream) -> Recorded {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        let n = stream.read(&mut chunk).unwrap();
        assert!(n > 0, "client closed before finishing its headers");
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap().to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let length: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .map(|(_, v)| v.parse().unwrap())
        .unwrap_or(0);
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < length {
        let n = stream.read(&mut chunk).unwrap();
        assert!(n > 0, "client closed before finishing its body");
        body.extend_from_slice(&chunk[..n]);
    }
    Recorded {
        request_line,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    }
}

fn write_response(stream: &mut TcpStream, canned: &Canned) {
    let mut out = format!(
        "HTTP/1.1 {} Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        canned.status,
        canned.body.len()
    );
    for (k, v) in &canned.headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("\r\n");
    out.push_str(&canned.body);
    let _ = stream.write_all(out.as_bytes());
    let _ = stream.flush();
}

/// OpenAI-shaped answer holding `vectors`, listed in `order` (to prove the
/// encoder sorts by `index`).
fn openai(vectors: &[Vec<f32>], order: &[usize]) -> String {
    let data: Vec<_> = order
        .iter()
        .map(|&i| serde_json::json!({"object": "embedding", "index": i, "embedding": vectors[i]}))
        .collect();
    serde_json::json!({"object": "list", "data": data}).to_string()
}

fn ollama(vectors: &[Vec<f32>]) -> String {
    serde_json::json!({"model": "m", "embeddings": vectors}).to_string()
}

fn fast(url: &str, dim: usize) -> HttpEncoder {
    HttpEncoder::new(url, "test-model", dim)
        .with_retry_backoff(Duration::from_millis(5))
        .with_timeout(Duration::from_secs(5))
}

#[test]
fn reads_openai_answers_in_index_order_and_sends_the_documented_request() {
    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    let mock = Mock::start(vec![ok(&openai(&vectors, &[2, 0, 1]))]);
    let encoder = fast(&mock.url, 2).with_api_key("sk-secret-123");

    let got = encoder.try_encode_batch(&["a", "b", "c"]).unwrap();
    assert_eq!(got, vectors);

    let seen = mock.requests();
    assert_eq!(seen.len(), 1);
    assert!(seen[0].request_line.starts_with("POST /v1/embeddings "));
    assert_eq!(seen[0].header("content-type"), Some("application/json"));
    assert_eq!(
        seen[0].header("authorization"),
        Some("Bearer sk-secret-123")
    );
    assert_eq!(
        seen[0].json(),
        serde_json::json!({"model": "test-model", "input": ["a", "b", "c"]})
    );
}

#[test]
fn reads_ollama_answers_and_needs_no_key() {
    let vectors = vec![vec![0.25, 0.75], vec![1.0, -1.0]];
    let mock = Mock::start(vec![ok(&ollama(&vectors))]);
    let encoder = fast(&mock.url, 2);

    assert_eq!(encoder.try_encode_batch(&["x", "y"]).unwrap(), vectors);
    assert_eq!(mock.requests()[0].header("authorization"), None);
}

#[test]
fn splits_large_inputs_into_batches_in_order() {
    let all: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, 1.0]).collect();
    let mock = Mock::start(vec![
        ok(&ollama(&all[0..2])),
        ok(&ollama(&all[2..4])),
        ok(&ollama(&all[4..5])),
    ]);
    let encoder = fast(&mock.url, 2).with_batch_size(2);

    let texts = ["t0", "t1", "t2", "t3", "t4"];
    assert_eq!(encoder.encode_batch(&texts), all);

    let seen = mock.requests();
    let inputs: Vec<_> = seen.iter().map(|r| r.json()["input"].clone()).collect();
    assert_eq!(
        inputs,
        vec![
            serde_json::json!(["t0", "t1"]),
            serde_json::json!(["t2", "t3"]),
            serde_json::json!(["t4"]),
        ]
    );
}

#[test]
fn an_empty_batch_makes_no_request() {
    let encoder = fast("http://127.0.0.1:9/never", 2);
    assert!(encoder.encode_batch(&[]).is_empty());
    assert_eq!(encoder.last_error(), None);
}

#[test]
fn retries_429_and_5xx_then_succeeds() {
    let v = vec![vec![1.0, 2.0]];
    let mock = Mock::start(vec![
        status(429, "slow down"),
        status(503, "busy"),
        ok(&ollama(&v)),
    ]);
    let encoder = fast(&mock.url, 2);

    assert_eq!(encoder.try_encode("hi").unwrap(), v[0]);
    assert_eq!(mock.requests().len(), 3);
}

#[test]
fn honours_retry_after() {
    let v = vec![vec![1.0, 2.0]];
    let mut limited = status(429, "slow down");
    limited.headers.push(("Retry-After", "1".to_string()));
    let mock = Mock::start(vec![limited, ok(&ollama(&v))]);
    let encoder = fast(&mock.url, 2);

    let started = Instant::now();
    encoder.try_encode("hi").unwrap();
    assert!(started.elapsed() >= Duration::from_millis(900));
}

#[test]
fn retries_are_bounded_and_the_last_status_is_reported() {
    let mock = Mock::start(vec![
        status(500, "boom 1"),
        status(500, "boom 2"),
        status(502, "boom 3"),
    ]);
    let encoder = fast(&mock.url, 2); // 2 retries -> 3 attempts

    match encoder.try_encode("hi") {
        Err(HttpEncoderError::Status {
            status, message, ..
        }) => {
            assert_eq!(status, 502);
            assert!(message.contains("boom 3"), "{message}");
        }
        other => panic!("expected a status error, got {other:?}"),
    }
    assert_eq!(mock.requests().len(), 3);
}

#[test]
fn retries_can_be_switched_off() {
    let mock = Mock::start(vec![status(503, "busy")]);
    let encoder = fast(&mock.url, 2).with_max_retries(0);
    assert!(encoder.try_encode("hi").is_err());
    assert_eq!(mock.requests().len(), 1);
}

#[test]
fn other_client_errors_are_not_retried() {
    let mock = Mock::start(vec![status(400, "bad model")]);
    let encoder = fast(&mock.url, 2);
    assert!(matches!(
        encoder.try_encode("hi"),
        Err(HttpEncoderError::Status { status: 400, .. })
    ));
    assert_eq!(mock.requests().len(), 1);
}

#[test]
fn the_api_key_never_appears_in_an_error_or_debug_output() {
    let key = "sk-very-secret-key";
    // A server that echoes the key back, as some do.
    let mock = Mock::start(vec![status(
        401,
        &format!("{{\"error\":\"Incorrect API key provided: {key}\"}}"),
    )]);
    let encoder = fast(&mock.url, 2).with_api_key(key);

    let err = encoder.try_encode("hi").unwrap_err().to_string();
    assert!(!err.contains(key), "{err}");
    assert!(err.contains("401"), "{err}");

    assert!(!format!("{encoder:?}").contains(key));
    assert_eq!(encoder.url(), mock.url);

    // The infallible path records the same redacted text.
    let mock = Mock::start(vec![status(401, key)]);
    let encoder = fast(&mock.url, 2).with_api_key(key);
    encoder.encode("hi");
    assert!(!encoder.last_error().unwrap().contains(key));
}

#[test]
fn a_url_with_credentials_or_a_query_is_not_echoed() {
    let encoder = HttpEncoder::new("http://user:pw@127.0.0.1:1/embed?api_key=abc123", "m", 2)
        .with_timeout(Duration::from_secs(2));
    let err = encoder.try_encode("hi").unwrap_err().to_string();
    assert!(!err.contains("abc123") && !err.contains("pw@"), "{err}");
    assert!(!format!("{encoder:?}").contains("abc123"));
}

#[test]
fn does_not_follow_redirects() {
    let mut redirect = status(307, "");
    redirect
        .headers
        .push(("Location", "http://127.0.0.1:9/elsewhere".to_string()));
    let mock = Mock::start(vec![redirect]);
    let encoder = fast(&mock.url, 2).with_api_key("sk-secret");
    assert!(matches!(
        encoder.try_encode("hi"),
        Err(HttpEncoderError::Status { status: 307, .. })
    ));
    assert_eq!(mock.requests().len(), 1);
}

#[test]
fn rejects_unusable_answers() {
    let cases: Vec<(&str, String)> = vec![
        ("not json", "<html>".to_string()),
        ("no embeddings key", "{\"hello\": 1}".to_string()),
        ("wrong count", ollama(&[vec![1.0, 2.0], vec![3.0, 4.0]])),
        ("wrong width", ollama(&[vec![1.0, 2.0, 3.0]])),
        ("empty vector", ollama(&[vec![]])),
        ("non-number", "{\"embeddings\": [[1.0, \"x\"]]}".to_string()),
        (
            "base64 string",
            "{\"data\": [{\"index\": 0, \"embedding\": \"AAAA\"}]}".to_string(),
        ),
    ];
    for (label, body) in cases {
        let mock = Mock::start(vec![ok(&body)]);
        let encoder = fast(&mock.url, 2);
        assert!(
            matches!(
                encoder.try_encode("hi"),
                Err(HttpEncoderError::Response { .. })
            ),
            "{label}"
        );
    }
}

#[test]
fn a_dead_endpoint_and_a_timeout_are_errors_not_hangs() {
    // Nothing listening.
    let dead = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/", dead.local_addr().unwrap());
    drop(dead);
    let encoder = HttpEncoder::new(&url, "m", 2).with_timeout(Duration::from_secs(2));
    assert!(matches!(
        encoder.try_encode("hi"),
        Err(HttpEncoderError::Transport { .. })
    ));

    // Accepts, then says nothing.
    let silent = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/", silent.local_addr().unwrap());
    let held = thread::spawn(move || {
        let (stream, _) = silent.accept().unwrap();
        thread::sleep(Duration::from_secs(2));
        drop(stream);
    });
    let encoder = HttpEncoder::new(&url, "m", 2).with_timeout(Duration::from_millis(300));
    let started = Instant::now();
    assert!(matches!(
        encoder.try_encode("hi"),
        Err(HttpEncoderError::Transport { .. })
    ));
    assert!(started.elapsed() < Duration::from_millis(1800));
    held.join().unwrap();
}

#[test]
fn failure_through_the_trait_is_an_empty_vector_a_store_refuses() {
    let mock = Mock::start(vec![status(400, "nope")]);
    let encoder = fast(&mock.url, 2);
    assert!(encoder.encode("hi").is_empty());
    assert!(encoder.last_error().unwrap().contains("400"));

    let mock = Mock::start(vec![status(400, "nope")]);
    let mut store = EmbeddingStore::new(Some(Box::new(fast(&mock.url, 2))));
    assert!(matches!(
        store.add(NodeId::new("n", "1"), "hi", None),
        Err(EmbeddingError::EmptyVector(_))
    ));
    assert_eq!(store.count(), 0);

    // A later success clears the error.
    let mock = Mock::start(vec![status(400, "nope"), ok(&ollama(&[vec![1.0, 2.0]]))]);
    let encoder = fast(&mock.url, 2);
    encoder.encode("a");
    assert!(encoder.last_error().is_some());
    assert_eq!(encoder.encode("b"), vec![1.0, 2.0]);
    assert_eq!(encoder.last_error(), None);
}

#[test]
fn reports_its_model_and_version_and_stamps_stored_vectors() {
    let mock = Mock::start(vec![ok(&ollama(&[vec![1.0, 0.0]]))]);
    let encoder = fast(&mock.url, 2).with_version("2024-05");
    assert_eq!(encoder.model_name(), "test-model");
    assert_eq!(encoder.version(), Some("2024-05"));
    assert_eq!(encoder.dimension(), 2);

    let mut store = EmbeddingStore::new(Some(Box::new(encoder)));
    store.add(NodeId::new("n", "1"), "hi", None).unwrap();
    assert_eq!(
        store.get(&NodeId::new("n", "1")).unwrap().model,
        "test-model"
    );
    let payload = store.export_embeddings();
    assert_eq!(payload.encoder_model.as_deref(), Some("test-model"));
    assert_eq!(payload.encoder_version.as_deref(), Some("2024-05"));
}

#[test]
fn probes_the_dimension_from_the_endpoint() {
    let mock = Mock::start(vec![ok(&ollama(&[vec![0.1, 0.2, 0.3]]))]);
    let encoder = HttpEncoder::new(&mock.url, "m", 1)
        .with_api_key("k")
        .probe_dimension()
        .unwrap();
    assert_eq!(encoder.dimension(), 3);

    let mock = Mock::start(vec![status(500, "x")]);
    let failed = HttpEncoder::new(&mock.url, "m", 1)
        .with_max_retries(0)
        .probe_dimension();
    assert!(failed.is_err());
}

#[test]
fn drives_a_semantic_graph_end_to_end() {
    // Two nodes, then a search: three requests with distinct, known vectors.
    let mock = Mock::start(vec![
        ok(&ollama(&[vec![1.0, 0.0]])),
        ok(&ollama(&[vec![0.0, 1.0]])),
        ok(&ollama(&[vec![1.0, 0.1]])),
    ]);
    let mut graph = SemanticGraph::new("g", Box::new(fast(&mock.url, 2)));
    graph
        .add_node(
            "doc",
            "1",
            vec![(
                "name",
                isongraph_vector_rs::PropertyValue::String("cats".into()),
            )],
            None,
        )
        .unwrap();
    graph
        .add_node(
            "doc",
            "2",
            vec![(
                "name",
                isongraph_vector_rs::PropertyValue::String("tax law".into()),
            )],
            None,
        )
        .unwrap();

    let hits = graph.similarity_search("kittens", 2, None, -1.0).unwrap();
    assert_eq!(hits[0].node_id, NodeId::new("doc", "1"));
    assert_eq!(
        graph.get_embedding(&NodeId::new("doc", "1")).unwrap().model,
        "test-model"
    );
}
