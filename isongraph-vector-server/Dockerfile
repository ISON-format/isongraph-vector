# Build from the REPOSITORY ROOT (the server depends on the sibling library directory, which is
# not on PyPI until 1.0.0 is published):
#
#   docker build -f isongraph-vector-server/Dockerfile -t igv-server .
#   docker build -f isongraph-vector-server/Dockerfile --build-arg WITH_SENTENCE_TRANSFORMERS=1 -t igv-server:st .
#
# The default image is small (no torch): use encoder.kind = "http" (an OpenAI/Ollama-style endpoint)
# or "mock" (tests only), or build the :st variant to run a local sentence-transformers model.
#
# Run:
#   docker run -v igv-data:/data -p 8080:8080 igv-server serve --config /etc/igv/config.json
#   docker run -v igv-data:/data igv-server tenant create acme
#   docker run -v igv-data:/data igv-server key create acme            # prints the key once
#
# This image speaks plain HTTP. Put a TLS-terminating proxy in front of it; see README "Threat model".

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin igv \
 && mkdir /data && chown igv:igv /data

WORKDIR /src
COPY pyproject.toml LICENSE README.md ./
COPY isongraph_vector-py ./isongraph_vector-py
COPY isongraph-vector-server ./isongraph-vector-server

ARG WITH_SENTENCE_TRANSFORMERS=0
# ison-graph (the library's graph engine) and numpy (vectorised search, ANN) are the only runtime
# dependencies; the library and the server are installed --no-deps so the library's hard
# sentence-transformers requirement is only pulled in when asked for.
RUN pip install ison-graph numpy \
 && pip install --no-deps . \
 && pip install --no-deps ./isongraph-vector-server \
 && if [ "$WITH_SENTENCE_TRANSFORMERS" = "1" ]; then pip install sentence-transformers; fi \
 && rm -rf /src

USER igv
WORKDIR /data
VOLUME /data
EXPOSE 8080

ENV IGV_DATA_DIR=/data \
    IGV_HOST=0.0.0.0 \
    IGV_PORT=8080

HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).status == 200 else 1)"

# SIGTERM (docker stop) triggers a graceful shutdown: stop accepting, drain, final snapshot.
STOPSIGNAL SIGTERM
ENTRYPOINT ["igv-server"]
CMD ["serve"]
