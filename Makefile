# aperdiff — Recognition of Semantic Differences on SwissGov-RSD.
#
# `make run` pulls the published multi-arch image and prints the bundled results.
# Everything else is selected with variables; ARGS is passed to the container
# verbatim, so any subcommand or flag the binary accepts is reachable.
#
#   make run
#   make run MODE=predict HEAD=unsupervised LANG=it TEST=1
#   make run MODE=predict INPUT=custom.jsonl OUTPUT=out JSON=1
#   make run MODE=results LANG=de JSON=1
#   make run MODE=tui
#   make run MODE=smoke SMOKE_FAST=1
#   make run MODE=speedtest
#   make run ARGS="results --lang it --json"
#   make build                       # build the image for the host architecture
#   make push                        # multi-arch (amd64 + arm64) build & push
#
# See README.md for the full invocation table.

.DEFAULT_GOAL := run

IMAGE       ?= ghcr.io/bitanath/rsd-swissgov:latest
LOCAL_IMAGE ?= aperdiff:local
PLATFORM    ?=

# Mode: results (default) | predict | tui | smoke | speedtest
MODE        ?=
# predict flags
HEAD        ?=
INPUT       ?=
OUTPUT      ?=
# shared flags
TEST        ?=
JSON        ?=
SMOKE_FAST  ?=
# shorthands
TUI         ?=
SMOKE       ?=
SPEEDTEST   ?=
# raw passthrough to the container (used verbatim, wins over everything else)
ARGS        ?=

# Host-TTY detection: paint the progress bar in place interactively, but keep
# clean periodic lines when stdout is piped/captured.
TTY := $(shell test -t 1 && echo -t)

# `LANG` is the POSIX locale variable, so it arrives pre-set (e.g. en_US.UTF-8).
# Only read it as a language selector when it looks like a plain code/list; the
# unambiguous `LANGS` always wins.
LANGS ?=
ifeq ($(strip $(LANGS)),)
  ifeq ($(findstring .,$(LANG))$(findstring _,$(LANG)),)
    LANG_SEL := $(LANG)
  endif
else
  LANG_SEL := $(LANGS)
endif

# --- default the mode from the shorthands -----------------------------------
ifeq ($(strip $(MODE)),)
  ifneq ($(strip $(TUI)),)
    MODE := tui
  else ifneq ($(strip $(SMOKE)),)
    MODE := smoke
  else ifneq ($(strip $(SPEEDTEST)),)
    MODE := speedtest
  else ifneq ($(strip $(HEAD)),)
    MODE := predict
  else
    MODE := results
  endif
endif

# --- assemble the flag list -------------------------------------------------
FLAGS :=
ifneq ($(strip $(HEAD)),)
FLAGS += --head $(HEAD)
endif
ifneq ($(strip $(LANG_SEL)),)
FLAGS += --lang $(LANG_SEL)
endif
ifneq ($(strip $(INPUT)),)
FLAGS += --input $(INPUT)
endif
ifneq ($(strip $(TEST)),)
FLAGS += --test
endif
ifneq ($(strip $(JSON)),)
FLAGS += --json
endif
ifneq ($(strip $(SMOKE_FAST)),)
FLAGS += --smoke-fast
endif

PREDICT_OUT := $(if $(strip $(OUTPUT)),$(OUTPUT),outputs/predictions)

# --- docker extras per mode -------------------------------------------------
ifeq ($(MODE),tui)
  DOCKER_EXTRA := -it --detach-keys="" -e TERM=xterm-256color
else
  DOCKER_EXTRA := $(TTY)
endif

# --- container argv: ARGS wins verbatim, else assemble from MODE + FLAGS ----
ifneq ($(strip $(ARGS)),)
  RUN_ARGS := $(ARGS)
else ifeq ($(MODE),tui)
  RUN_ARGS :=
else ifeq ($(MODE),smoke)
  RUN_ARGS := --smoke $(FLAGS)
else ifeq ($(MODE),speedtest)
  RUN_ARGS := --speedtest
else ifeq ($(MODE),predict)
  RUN_ARGS := predict --output $(PREDICT_OUT) $(FLAGS)
else ifeq ($(MODE),results)
  RUN_ARGS := results $(FLAGS)
else
  $(error unknown MODE=$(MODE) — use results | predict | tui | smoke | speedtest)
endif

PLATFORM_ARG := $(if $(strip $(PLATFORM)),--platform $(PLATFORM),)
ENV_ARG      := -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY
MOUNT        := -v "$(CURDIR):/data"

.PHONY: run build push help

run:
	docker run --rm $(DOCKER_EXTRA) $(PLATFORM_ARG) $(ENV_ARG) $(MOUNT) $(IMAGE) $(RUN_ARGS)

build:
	docker build -t $(LOCAL_IMAGE) .

push:
	docker buildx build --platform linux/amd64,linux/arm64 -t $(IMAGE) --push .

help:
	@sed -n '2,20p' Makefile
