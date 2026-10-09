# Vendored: Hunk OpenTUI diff renderer

- Source: https://github.com/modem-dev/hunk
- Commit: `9566e33d930e8844a0fe910e7b9d0f822b0728cb`
- Package: `hunkdiff` 0.23.0
- License: MIT (Modem Labs Inc.) — see upstream `LICENSE`
- Scope: the 105-file transitive closure of the OpenTUI diff components, copied
  from `packages/hunk/src/` with the same relative layout so imports resolve.

Refresh with `scripts/vendor-hunk.sh` (optionally pass a path to a Hunk checkout).

## Local modifications (keep in sync)

Current:

- `vendor/hunk-vcs-source.ts` — local shim for the `@hunk/vcs/source` workspace
  import in `core/changeset/fileSource.ts`; resolved via tsconfig `paths`.
- `core/run/version.ts` — inlined the `hunkdiff` version constant; the upstream
  `../../../package.json` import assumes the extra `src/` path level we drop when
  vendoring.

Applied (phase 2 — semantic layer):

- `core/changeset/model.ts` — `DiffFile.note?: string`.
- `opentui/types.ts`, `opentui/model.ts` — public `HunkDiffFileInput.note?` and
  threading it into the internal file.
- `ui/diff/diffRowModel.ts` — added a `note` row variant.
- `ui/diff/diffRows.ts` — emit the note row and apply our own word-diff overlay
  (`@lib/worddiff`) in place of Pierre's `data-diff-span` emphasis.
- `ui/diff/reviewRenderPlan.ts` — planned-row case, stable key, guard, and
  anchor exclusion for the `note` row.
- `ui/diff/DiffRowView.tsx` — dispatch `note` rows to `@interface/note-row`.

Still deferred:

- `opentui/*` per-side `leftLanguage` / `rightLanguage` for the one-file,
  two-language mode.
