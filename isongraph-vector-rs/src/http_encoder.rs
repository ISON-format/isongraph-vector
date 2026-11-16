//! An `EmbeddingEncoder` that asks an HTTP embeddings endpoint.
//!
//! Enabled by the `http-encoder` feature (off by default; the default build
//! has no HTTP client). **Every text you embed or search with is sent to the
//! configured URL**, so point it only at a service you are willing to give that
//! text to.
//!
//! The request is `POST <url>` with a JSON body `{"model": ..., "input":
//! [text, ...]}`, which both of the common services accept:
//!
//! * OpenAI-compatible (`/v1/embeddings`): the answer's
//!   `data[i].embedding` are read in `index` order.
//! * Ollama (`/api/embed`): the answer's `embeddings` array.
//!
//! The shape is recognised from the answer, not from the URL.
//!
//! ```no_run
//! use isongraph_vector_rs::{EmbeddingStore, HttpEncoder};
//!
//! let encoder = HttpEncoder::new("http://localhost:11434/api/embed", "nomic-embed-text", 768)
//!     .with_batch_size(32);
//! let mut store = EmbeddingStore::new(Some(Box::new(encoder)));
//! ```
//!
//! # Failures
//!
//! `EmbeddingEncoder` methods cannot return an error, so a failed request
//! makes `encode` / `encode_batch` return *empty* vectors. Every store rejects
//! an empty vector (`EmbeddingError::EmptyVector`) rather than storing it, so
//! nothing is ever written from a failed call. The cause is kept in
//! `last_error()` and printed to stderr; `try_encode` and `try_encode_batch`
//! return it directly.

use std::io::Read;
use std::sync::Mutex;
use std::time::Duration;

use serde_json::Value as Json;
use thiserror::Error;

use crate::EmbeddingEncoder;

/// Largest response body read, so a misbehaving endpoint cannot exhaust memory.
const MAX_RESPONSE_BYTES: u64 = 256 * 1024 * 1024;
/// How much of an error response is quoted in an error message.
const ERROR_SNIPPET_CHARS: usize = 200;
/// Longest wait honoured from a `Retry-After` header.
const MAX_RETRY_WAIT: Duration = Duration::from_secs(30);

/// Why a request to the embeddings endpoint failed. Never contains the API key.
#[derive(Error, Debug)]
pub enum HttpEncoderError {
    /// The request did not complete (connection refused, DNS, TLS, timeout).
    #[error("embedding request to {url} failed: {message}")]
    Transport { url: String, message: String },
    /// The endpoint answered with an error status.
    #[error("embedding endpoint {url} returned HTTP {status}: {message}")]
    Status {
        url: String,
        status: u16,
        message: String,
    },
    /// The endpoint answered, but not with usable embeddings.
    #[error("embedding endpoint {url} sent an unusable response: {message}")]
    Response { url: String, message: String },
}

/// Encoder backed by an HTTP embeddings endpoint. See the module docs.
pub struct HttpEncoder {
    url: String,
    model: String,
    dimension: usize,
    api_key: Option<String>,
    batch_size: usize,
    timeout: Duration,
    version: Option<String>,
    max_retries: u32,
    retry_backoff: Duration,
    agent: ureq::Agent,
    last_error: Mutex<Option<String>>,
}

impl std::fmt::Debug for HttpEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpEncoder")
            .field("url", &safe_url(&self.url))
            .field("model", &self.model)
            .field("dimension", &self.dimension)
            .field("api_key", &self.api_key.as_ref().map(|_| "<set>"))
            .field("batch_size", &self.batch_size)
            .field("timeout", &self.timeout)
            .field("version", &self.version)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl HttpEncoder {
    /// An encoder for `model` at `url`, whose vectors are `dimension` wide.
    ///
    /// The dimension is stated rather than discovered so that building the
    /// encoder does no network I/O; a response of any other width is an error.
    /// Use [`probe_dimension`](Self::probe_dimension) to learn it from the
    /// endpoint instead.
    ///
    /// Defaults: no API key, batches of 32 texts, a 30 second timeout, two
    /// retries (so up to three attempts) on HTTP 429 and 5xx.
    pub fn new(url: impl Into<String>, model: impl Into<String>, dimension: usize) -> Self {
        assert!(dimension > 0, "dimension must be positive");
        let timeout = Duration::from_secs(30);
        Self {
            url: url.into(),
            model: model.into(),
            dimension,
            api_key: None,
            batch_size: 32,
            timeout,
            version: None,
            max_retries: 2,
            retry_backoff: Duration::from_millis(250),
            agent: build_agent(timeout),
            last_error: Mutex::new(None),
        }
    }

    /// Send `Authorization: Bearer <key>` with every request. The key is never
    /// included in an error message or in `Debug` output.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        let key = api_key.into();
        self.api_key = if key.is_empty() { None } else { Some(key) };
        self
    }

    /// Texts per request (at least 1). Larger inputs are split across requests.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Give up on a request after this long (connecting and reading together).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.agent = build_agent(timeout);
        self
    }

    /// Report an encoder/model version, recorded in the sidecar.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// How many times a request answered with HTTP 429 or 5xx is repeated
    /// (default 2; 0 disables retrying).
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// First wait before a retry (default 250 ms); each further retry doubles
    /// it. A `Retry-After` header, when present, replaces it (capped at 30 s).
    pub fn with_retry_backoff(mut self, backoff: Duration) -> Self {
        self.retry_backoff = backoff;
        self
    }

    /// Replace the configured dimension with the width the endpoint actually
    /// returns, by embedding one short text.
    pub fn probe_dimension(mut self) -> Result<Self, HttpEncoderError> {
        let saved = self.dimension;
        // Accept any width for the probe itself.
        self.dimension = 0;
        let probed = self.request_batch(&["dimension probe"]);
        self.dimension = saved;
        let width = probed?.first().map(Vec::len).unwrap_or(0);
        if width == 0 {
            return Err(HttpEncoderError::Response {
                url: safe_url(&self.url),
                message: "the probe returned no vector".to_string(),
            });
        }
        self.dimension = width;
        Ok(self)
    }

    /// The URL texts are sent to, without any credentials or query string.
    pub fn url(&self) -> String {
        safe_url(&self.url)
    }

    /// The cause of the most recent failed `encode` / `encode_batch`, cleared
    /// by the next success.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|e| e.clone())
    }

    /// Embed one text, reporting failure instead of returning an empty vector.
    pub fn try_encode(&self, text: &str) -> Result<Vec<f32>, HttpEncoderError> {
        let mut vectors = self.try_encode_batch(&[text])?;
        Ok(vectors.pop().unwrap_or_default())
    }

    /// Embed many texts, `batch_size` per request, in input order.
    pub fn try_encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HttpEncoderError> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            out.extend(self.request_batch(chunk)?);
        }
        Ok(out)
    }

    fn remember(&self, error: Option<String>) {
        if let Ok(mut slot) = self.last_error.lock() {
            *slot = error;
        }
    }

    /// One batch, with the retry policy.
    fn request_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HttpEncoderError> {
        let body = serde_json::json!({ "model": self.model, "input": texts }).to_string();
        let mut attempt = 0u32;
        loop {
            match self.send_once(&body) {
                Ok(text) => return self.parse(&text, texts.len()),
                Err(Attempt::Retryable { retry_after, .. }) if attempt < self.max_retries => {
                    let wait = retry_after
                        .unwrap_or_else(|| self.retry_backoff * 2u32.saturating_pow(attempt))
                        .min(MAX_RETRY_WAIT);
                    std::thread::sleep(wait);
                    attempt += 1;
                }
                Err(Attempt::Retryable { error, .. }) | Err(Attempt::Fatal(error)) => {
                    return Err(error)
                }
            }
        }
    }

    fn send_once(&self, body: &str) -> Result<String, Attempt> {
        let mut request = self
            .agent
            .post(&self.url)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json");
        if let Some(key) = &self.api_key {
            request = request.set("Authorization", &format!("Bearer {key}"));
        }
        match request.send_string(body) {
            Ok(response) if !(200..300).contains(&response.status()) => {
                // With redirects off ureq hands a 3xx back as a success.
                Err(Attempt::Fatal(HttpEncoderError::Status {
                    url: safe_url(&self.url),
                    status: response.status(),
                    message: "redirects are not followed".to_string(),
                }))
            }
            Ok(response) => {
                let mut text = String::new();
                response
                    .into_reader()
                    .take(MAX_RESPONSE_BYTES)
                    .read_to_string(&mut text)
                    .map_err(|e| {
                        Attempt::Fatal(self.transport_error(&format!("reading the response: {e}")))
                    })?;
                Ok(text)
            }
            Err(ureq::Error::Status(status, response)) => {
                let retry_after = response
                    .header("Retry-After")
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .map(Duration::from_secs);
                let mut snippet = String::new();
                let _ = response
                    .into_reader()
                    .take(4096)
                    .read_to_string(&mut snippet);
                let error = HttpEncoderError::Status {
                    url: safe_url(&self.url),
                    status,
                    message: self.redact(&snippet_of(&snippet)),
                };
                if status == 429 || status >= 500 {
                    Err(Attempt::Retryable { error, retry_after })
                } else {
                    Err(Attempt::Fatal(error))
                }
            }
            Err(ureq::Error::Transport(t)) => {
                let detail = match t.message() {
                    Some(message) => format!("{:?}: {message}", t.kind()),
                    None => format!("{:?}", t.kind()),
                };
                Err(Attempt::Fatal(self.transport_error(&detail)))
            }
        }
    }

    fn transport_error(&self, detail: &str) -> HttpEncoderError {
        HttpEncoderError::Transport {
            url: safe_url(&self.url),
            message: self.redact(detail),
        }
    }

    fn bad_response(&self, message: String) -> HttpEncoderError {
        HttpEncoderError::Response {
            url: safe_url(&self.url),
            message: self.redact(&message),
        }
    }

    /// Replace the API key, should anything quote it back, with `[redacted]`.
    fn redact(&self, text: &str) -> String {
        match &self.api_key {
            Some(key) => text.replace(key.as_str(), "[redacted]"),
            None => text.to_string(),
        }
    }

    /// Read OpenAI (`data[].embedding`, ordered by `index`) or Ollama
    /// (`embeddings`) out of a response, checking count and width.
    fn parse(&self, text: &str, expected: usize) -> Result<Vec<Vec<f32>>, HttpEncoderError> {
        let json: Json =
            serde_json::from_str(text).map_err(|e| self.bad_response(format!("not JSON ({e})")))?;

        let raw: Vec<&Json> = if let Some(data) = json.get("data").and_then(Json::as_array) {
            let mut items: Vec<(usize, &Json)> = data
                .iter()
                .enumerate()
                .map(|(position, item)| {
                    let index = item
                        .get("index")
                        .and_then(Json::as_u64)
                        .map(|i| i as usize)
                        .unwrap_or(position);
                    (index, item.get("embedding").unwrap_or(&Json::Null))
                })
                .collect();
            items.sort_by_key(|(index, _)| *index);
            items.into_iter().map(|(_, v)| v).collect()
        } else if let Some(embeddings) = json.get("embeddings").and_then(Json::as_array) {
            embeddings.iter().collect()
        } else {
            return Err(self.bad_response(
                "no `data[].embedding` (OpenAI) or `embeddings` (Ollama) in the answer".to_string(),
            ));
        };

        if raw.len() != expected {
            return Err(self.bad_response(format!(
                "{} embeddings for {} texts",
                raw.len(),
                expected
            )));
        }
        let mut vectors = Vec::with_capacity(raw.len());
        for (i, value) in raw.into_iter().enumerate() {
            let numbers = value
                .as_array()
                .ok_or_else(|| self.bad_response(format!("embedding {i} is not an array")))?;
            let vector: Vec<f32> = numbers
                .iter()
                .map(|n| n.as_f64().map(|f| f as f32))
                .collect::<Option<_>>()
                .ok_or_else(|| self.bad_response(format!("embedding {i} holds a non-number")))?;
            if vector.is_empty() {
                return Err(self.bad_response(format!("embedding {i} is empty")));
            }
            if self.dimension != 0 && vector.len() != self.dimension {
                return Err(self.bad_response(format!(
                    "embedding {i} has {} dimensions, expected {}",
                    vector.len(),
                    self.dimension
                )));
            }
            vectors.push(vector);
        }
        Ok(vectors)
    }
}

enum Attempt {
    Retryable {
        error: HttpEncoderError,
        retry_after: Option<Duration>,
    },
    Fatal(HttpEncoderError),
}

fn build_agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(timeout)
        // Never follow a redirect: the API key would travel to wherever it
        // points, and a POST body is not safe to replay blindly.
        .redirects(0)
        .build()
}

/// The URL minus any `user:password@` and `?query`, safe to put in a message.
fn safe_url(url: &str) -> String {
    let (scheme, rest) = match url.split_once("://") {
        Some((scheme, rest)) => (Some(scheme), rest),
        None => (None, url),
    };
    let rest = rest.split(['?', '#']).next().unwrap_or("");
    let (authority, path) = match rest.find('/') {
        Some(i) => rest.split_at(i),
        None => (rest, ""),
    };
    let host = authority.rsplit('@').next().unwrap_or(authority);
    match scheme {
        Some(scheme) => format!("{scheme}://{host}{path}"),
        None => format!("{host}{path}"),
    }
}

fn snippet_of(body: &str) -> String {
    let flat: String = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > ERROR_SNIPPET_CHARS {
        let cut: String = flat.chars().take(ERROR_SNIPPET_CHARS).collect();
        format!("{cut}...")
    } else if flat.is_empty() {
        "(empty body)".to_string()
    } else {
        flat
    }
}

impl EmbeddingEncoder for HttpEncoder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    /// The model sent in every request.
    fn model_name(&self) -> &str {
        &self.model
    }

    fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    /// An empty vector if the request fails; see the module docs.
    fn encode(&self, text: &str) -> Vec<f32> {
        match self.try_encode(text) {
            Ok(vector) => {
                self.remember(None);
                vector
            }
            Err(e) => {
                self.report(&e);
                Vec::new()
            }
        }
    }

    /// One empty vector per text if a request fails; see the module docs.
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        match self.try_encode_batch(texts) {
            Ok(vectors) => {
                self.remember(None);
                vectors
            }
            Err(e) => {
                self.report(&e);
                vec![Vec::new(); texts.len()]
            }
        }
    }
}

impl HttpEncoder {
    fn report(&self, error: &HttpEncoderError) {
        let message = error.to_string();
        eprintln!("warning: {message}");
        self.remember(Some(message));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_url_drops_credentials_and_query() {
        assert_eq!(
            safe_url("https://user:pw@host:8080/v1/embeddings?key=abc#frag"),
            "https://host:8080/v1/embeddings"
        );
        assert_eq!(safe_url("http://localhost:11434"), "http://localhost:11434");
        assert_eq!(safe_url("localhost/x?y"), "localhost/x");
    }

    #[test]
    fn snippets_are_flattened_and_bounded() {
        assert_eq!(snippet_of(""), "(empty body)");
        assert_eq!(snippet_of("a\n  b"), "a b");
        assert!(snippet_of(&"x".repeat(500)).chars().count() <= ERROR_SNIPPET_CHARS + 3);
    }
}
