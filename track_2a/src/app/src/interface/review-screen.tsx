import { useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { AutocompleteOption } from "../components/autocomplete";
import { Alert } from "../components/alert";
import { Spinner } from "../components/spinner";
import type { Head, Signal } from "../libraries/analyze";
import { signalFromGold } from "../libraries/analyze";
import { COMMANDS, runCommand, type LayoutMode, type ModalKind } from "../libraries/commands";
import { findDoc, type Lang, type Split } from "../libraries/dataset";
import type { PairScores } from "../funcs";
import { useHunkTheme } from "../libraries/theme";
import { createAnalysisClient, type Phase } from "../progress";
import { buildTree, DocNav, firstDocId, langId, splitId, visibleRows } from "./doc-nav";
import { HelpModal } from "./help-modal";
import { NotePanel, PairView } from "./pair-view";
import { PromptBar } from "./prompt-bar";
import { ThemeModal } from "./theme-modal";
import { useToast } from "./toaster";

const NAV_WIDTH = 34;

interface DocCache {
  head: Head | null;
  scores: PairScores | null;
  note: string;
  signal: Signal | null;
}

const COMMAND_OPTIONS: AutocompleteOption[] = COMMANDS.map((command) => ({
  name: command.name,
  label: command.label,
  description: command.description,
}));

function toggleMember(set: ReadonlySet<string>, id: string, open?: boolean): Set<string> {
  const next = new Set(set);
  if (open ?? !next.has(id)) next.add(id);
  else next.delete(id);
  return next;
}

function phaseLabel(phase: Phase | null, pass: number | undefined, progress: number): string {
  const pct = progress > 0 ? ` · ${Math.round(progress * 100)}%` : "";
  switch (phase) {
    case "loading":
      return "loading model…";
    case "capturing":
      return `capturing layers ${pass ?? 1}/2${pct}`;
    case "classifying":
      return `classifying intermediate outputs${pct}`;
    case "writing":
      return `writing note${pct}`;
    default:
      return "working…";
  }
}

export function ReviewScreen(_props: { initialFile?: unknown }) {
  const { width } = useTerminalDimensions();
  const { name, theme } = useHunkTheme();
  const renderer = useRenderer();
  const toast = useToast();

  const client = useMemo(() => createAnalysisClient(), []);
  const tree = useMemo(() => buildTree(), []);

  const [selectedId, setSelectedId] = useState(() => firstDocId(tree));
  const [layout, setLayout] = useState<LayoutMode>("split");
  const [modal, setModal] = useState<ModalKind | null>(null);
  const [showHighlights, setShowHighlights] = useState(true);
  const [showGold, setShowGold] = useState(true);
  const [status, setStatus] = useState("idle");
  const [phase, setPhase] = useState<Phase | null>(null);
  const [pass, setPass] = useState<number | undefined>(undefined);
  const [progress, setProgress] = useState(0);
  const [head, setHead] = useState<Head | null>(null);
  const [scores, setScores] = useState<PairScores | null>(null);
  const [note, setNote] = useState("");
  const [signal, setSignal] = useState<Signal | null>(null);
  const [running, setRunning] = useState(false);
  const [expandedSplits, setExpandedSplits] = useState<ReadonlySet<string>>(() => new Set([splitId("test")]));
  const [expandedLangs, setExpandedLangs] = useState<ReadonlySet<string>>(new Set());
  const [cursorIndex, setCursorIndex] = useState(0);
  const [promptText, setPromptText] = useState("");

  const rows = useMemo(
    () => visibleRows(tree, expandedSplits, expandedLangs),
    [tree, expandedSplits, expandedLangs],
  );

  const cache = useRef<Record<string, DocCache>>({});
  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;

  const doc = findDoc(selectedId);
  const mainWidth = Math.max(40, width - NAV_WIDTH);

  const goldSignal = doc ? signalFromGold(doc) : null;
  const goldVerdict =
    goldSignal === null
      ? null
      : goldSignal === "same"
        ? "The docs are semantically the same"
        : "The docs are semantically different at the highlighted word pairs";
  const activeSignal = showGold ? goldSignal : signal;

  useEffect(() => {
    client.cancel();
    const c = cache.current[selectedId];
    setHead(c?.head ?? null);
    setScores(c?.scores ?? null);
    setNote(c?.note ?? "");
    setSignal(c?.signal ?? null);
    setStatus("idle");
    setPhase(null);
    setPass(undefined);
    setProgress(0);
    setRunning(false);
  }, [selectedId, client]);

  useEffect(() => () => client.dispose(), [client]);

  const quit = useCallback(() => {
    client.dispose();
    renderer.destroy();
    process.exit(0);
  }, [renderer, client]);

  const analyse = useCallback(
    async (target: Head) => {
      if (!doc) return;
      const id = doc.id;
      setRunning(true);
      setPhase("loading");
      setPass(undefined);
      setProgress(0);
      setHead(target);
      setScores(cache.current[id]?.head === target ? cache.current[id]?.scores ?? null : null);
      setStatus(target === "gru" ? "computing layerwise + GRU…" : "computing layerwise + DiffAlign…");

      const result = await client.analyse(doc, target, {
        onPhase: (next, nextPass) => {
          if (selectedRef.current !== id) return;
          setPhase(next);
          setPass(nextPass);
          if (next !== "capturing") setProgress(0);
        },
        onProgress: (next, fraction, nextPass) => {
          if (selectedRef.current !== id) return;
          setPhase(next);
          setPass(nextPass);
          setProgress(fraction);
        },
        onScores: (a, b, sig) => {
          if (selectedRef.current !== id) return;
          const scored: PairScores = { a, b };
          setScores(scored);
          setSignal(sig);
          // Model output is in: switch off ground truth so the streamed note + scores show.
          setShowGold(false);
          cache.current[id] = { head: target, scores: scored, note: cache.current[id]?.note ?? "", signal: sig };
        },
        onNote: (text) => {
          if (selectedRef.current === id) setNote(text);
        },
        onNoteProgress: (fraction) => {
          if (selectedRef.current === id) setProgress(fraction);
        },
      });

      if (selectedRef.current === id) {
        setScores(result.scores);
        setSignal(result.signal);
        setNote(result.note);
        if (result.scores) {
          cache.current[id] = { head: target, scores: result.scores, note: result.note, signal: result.signal };
        }
        setRunning(false);
        setPhase(null);
        setPass(undefined);
        setProgress(0);
        setStatus("idle");
      }
    },
    [doc, client],
  );

  const selectDoc = useCallback((docId: string) => setSelectedId(docId), []);
  const toggleSplit = useCallback(
    (split: Split) => setExpandedSplits((set) => toggleMember(set, splitId(split))),
    [],
  );
  const toggleLang = useCallback(
    (split: Split, lang: Lang) => setExpandedLangs((set) => toggleMember(set, langId(split, lang))),
    [],
  );

  const safeCursor = Math.min(cursorIndex, Math.max(0, rows.length - 1));

  useKeyboard((key) => {
    if (key.ctrl) {
      if (key.name === "c") quit();
      else if (key.name === "t") setModal("theme");
      else if (key.name === "p") setModal("help");
      else if (key.name === "r") void analyse("gru");
      else if (key.name === "k") void analyse("diffalign");
      else if (key.name === "e") setShowHighlights((h) => !h);
      else if (key.name === "g") setShowGold((g) => !g);
      return;
    }
    if (modal !== null) return;
    // While a command is being typed the prompt owns the keys.
    if (promptText.trim().length > 0) return;

    const row = rows[safeCursor];
    const openRow = () => {
      if (!row) return;
      if (row.kind === "split") toggleSplit(row.split);
      else if (row.kind === "lang") toggleLang(row.split, row.lang);
      else selectDoc(row.docId);
    };

    if (key.name === "up") {
      key.preventDefault();
      setCursorIndex(Math.max(0, safeCursor - 1));
    } else if (key.name === "down") {
      key.preventDefault();
      setCursorIndex(Math.min(rows.length - 1, safeCursor + 1));
    } else if (key.name === "right" || key.name === "space" || key.name === "return") {
      key.preventDefault();
      openRow();
    } else if (key.name === "left") {
      key.preventDefault();
      if (!row) return;
      if (row.kind === "split") setExpandedSplits((set) => toggleMember(set, row.id, false));
      else if (row.kind === "lang") setExpandedLangs((set) => toggleMember(set, row.id, false));
    }
  });

  const onSubmit = (text: string) => {
    runCommand(text, {
      layout,
      setLayout,
      quit,
      openModal: setModal,
      toast: (message, variant) => toast(message, variant ? { variant } : undefined),
      runGru: () => void analyse("gru"),
      runDiffalign: () => void analyse("diffalign"),
      toggleHighlights: () => setShowHighlights((h) => !h),
      toggleGold: () => setShowGold((g) => !g),
    });
  };

  return (
    <box
      style={{
        width: "100%",
        height: "100%",
        flexDirection: "column",
        backgroundColor: theme.background,
      }}
    >
      <box style={{ width: "100%", flexShrink: 0, flexDirection: "column" }}>
        <Alert
          variant="info"
          title={`aperdiff · RSD · ${name}-theme`}
          message={running ? undefined : `${status}${doc ? ` · ${doc.id}` : ""}`}
        >
          {running ? (
            <box flexDirection="row" gap={1}>
              <Spinner type="dots" color={theme.accent} />
              <text fg={theme.muted}>{phaseLabel(phase, pass, progress)}</text>
            </box>
          ) : null}
        </Alert>
      </box>

      <box
        style={{
          flexGrow: 1,
          flexDirection: "row",
          minHeight: 0,
          backgroundColor: theme.background,
        }}
      >
        <box style={{ width: NAV_WIDTH, height: "100%", backgroundColor: theme.panel }}>
          <scrollbox width="100%" height="100%" scrollY>
            <DocNav
              rows={rows}
              selectedId={selectedId}
              cursorIndex={safeCursor}
              width={NAV_WIDTH}
              theme={theme}
              onToggleSplit={toggleSplit}
              onToggleLang={toggleLang}
              onSelect={selectDoc}
            />
          </scrollbox>
        </box>
        <box style={{ width: 2, height: "100%", backgroundColor: theme.background }} />
        <box
          style={{
            width: mainWidth,
            height: "100%",
            flexDirection: "column",
            backgroundColor: theme.background,
          }}
        >
          <scrollbox
            key={selectedId}
            width="100%"
            flexGrow={1}
            minHeight={0}
            scrollY
            viewportCulling
          >
            {doc ? (
              <PairView
                doc={doc}
                width={mainWidth}
                theme={theme}
                scores={scores}
                head={head}
                signal={activeSignal}
                showHighlights={showHighlights}
                showGold={showGold}
              />
            ) : (
              <text fg={theme.muted}>No document selected.</text>
            )}
          </scrollbox>
          <NotePanel
            note={note}
            verdict={showGold ? goldVerdict : null}
            running={running}
            width={mainWidth}
            theme={theme}
          />
        </box>
      </box>

      <PromptBar
        options={COMMAND_OPTIONS}
        onSubmit={onSubmit}
        onChange={setPromptText}
        focused={modal === null}
      />

      <box style={{ width: "100%", height: 1, flexDirection: "row", flexShrink: 0 }}>
        <text
          fg={theme.muted}
        >{`[↑/↓] doc  ·  [←/→] collapse/expand  ·  [^e] highlights  ·  [^g] ground truth  ·  [^r] GRU  ·  [^k] DiffAlign  ·  [^p] help  ·  [^t] theme  ·  [^c] quit`}</text>
      </box>

      {modal === "help" && <HelpModal onClose={() => setModal(null)} />}
      {modal === "theme" && <ThemeModal onClose={() => setModal(null)} />}
    </box>
  );
}
