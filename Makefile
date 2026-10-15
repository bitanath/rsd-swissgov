# aperdiff — `make run` pulls the published image and prints the bundled results.
# Vars: MODE=results|predict|tui|smoke|speedtest, HEAD, LANG (or LANGS), INPUT,
# OUTPUT, TEST, JSON, SMOKE_FAST, ARGS, IMAGE, PLATFORM. `make build`/`push` build/publish.

.DEFAULT_GOAL := run

IMAGE       ?= ghcr.io/bitanath/rsd-swissgov:latest
LOCAL_IMAGE ?= aperdiff:local
PLATFORM    ?=

MODE        ?=
HEAD        ?=
INPUT       ?=
OUTPUT      ?=
TEST        ?=
JSON        ?=
SMOKE_FAST  ?=
TUI         ?=
SMOKE       ?=
SPEEDTEST   ?=
ARGS        ?=

TTY := $(shell test -t 1 && echo -t)

LANGS ?=
ifeq ($(strip $(LANGS)),)
  ifeq ($(findstring .,$(LANG))$(findstring _,$(LANG)),)
    LANG_SEL := $(LANG)
  endif
else
  LANG_SEL := $(LANGS)
endif

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

ifeq ($(MODE),tui)
  DOCKER_EXTRA := -it --detach-keys="" -e TERM=xterm-256color
else
  DOCKER_EXTRA := $(TTY)
endif

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
	@sed -n '1,3p' Makefile
