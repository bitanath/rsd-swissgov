# aperdiff — multi-stage build from the OpenTUI (Bun/TypeScript) app, the
# prebuilt CPU engine libraries, and the Apertus GGUF.
#
# Every RUN executes on the *build* platform (native); the target stages are
# COPY-only, so `docker buildx build --platform linux/amd64,linux/arm64` needs
# no QEMU emulation. On Apple Silicon the container runtime (colima + Rosetta)
# serves both base-image architectures.
#
#   docker build -t aperdiff:local .
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -t ghcr.io/bitanath/rsd-swissgov:latest --push .

# ---- builder: compile the Bun/OpenTUI binary (native on the build platform) --
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

# ---- model: fetch the Apertus Q5_K_M GGUF (architecture-independent) --------
FROM --platform=$BUILDPLATFORM debian:stable-slim AS model
ARG MODEL_URL=https://huggingface.co/justacoderwhocodes/apertus-1.5-text-only/resolve/main/apertus-v1.5-8b-text-Q5_K_M.gguf
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /models; \
    curl -fL "$MODEL_URL" -o /models/apertus.gguf

# ---- runtime: target platform, COPY only (no RUN → no emulation) ------------
FROM debian:stable-slim AS runtime
COPY --from=builder /out/aperdiff /usr/local/bin/aperdiff
COPY --from=model /models/apertus.gguf /opt/apertus.gguf
# The engine loads a standalone GGUF via ELTHUR_MODEL when no model is appended.
ENV ELTHUR_MODEL=/opt/apertus.gguf
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/aperdiff"]
