# The canonical Makefile lives at the repository root. This forwards so `make run`
# (and the other targets) also work from inside track_2a/. Command-line variables
# such as MODE=, HEAD=, LANG= are passed through to the sub-make automatically.
.DEFAULT_GOAL := run

.PHONY: run build push help

run build push help:
	@$(MAKE) --no-print-directory -C .. $@
