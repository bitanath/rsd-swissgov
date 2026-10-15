# Aperdiff — multi-stage build: OpenTUI app + prebuilt engine libs + Apertus GGUF.
# RUN steps execute on the build platform (native); target stages are COPY-only,
# so `docker buildx build --platform linux/amd64,linux/arm64` needs no emulation.

FROM --platform=$BUILDPLATFORM oven/bun:1 AS builder
ARG TARGETARCH
WORKDIR /app
COPY track_2a/src/app/ ./
COPY track_2a/src/libelthur-${TARGETARCH}.so ./libelthur.bin
RUN set -eux; \
    bun install --os="*" --cpu="*"; \
    case "$TARGETARCH" in \
      amd64) BUN_TARGET=bun-linux-x64 ;; \
      arm64) BUN_TARGET=bun-linux-arm64 ;; \
      *) echo "unsupported TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac; \
    bun build --compile index.tsx src/libraries/progress/worker.ts \
      --target "$BUN_TARGET" --outfile /out/aperdiff

FROM --platform=$BUILDPLATFORM debian:stable-slim AS model
ARG MODEL_URL=https://huggingface.co/justacoderwhocodes/apertus-1.5-text-only/resolve/main/apertus-v1.5-8b-text-Q5_K_M.gguf
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /models; \
    curl -fL "$MODEL_URL" -o /models/apertus.gguf

FROM debian:stable-slim AS runtime
COPY --from=builder /out/aperdiff /usr/local/bin/aperdiff
COPY --from=model /models/apertus.gguf /opt/apertus.gguf
ENV ELTHUR_MODEL=/opt/apertus.gguf
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/aperdiff"]
