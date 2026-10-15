import { useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { AutocompleteOption } from "../components/autocomplete";
import { Alert } from "../components/alert";
import { Spinner } from "../components/spinner";
import type { Head, Signal } from "../libraries/clit/analyze";
import { HEAD_LABEL, signalFromGold, type Combined } from "../libraries/clit/analyze";
import { COMMANDS, runCommand, type ModalKind } from "../libraries/clit/commands";
import { findDoc, type Lang, type Split } from "../libraries/clit/dataset";
import { useCopyOnSelect } from "../libraries/clit/clipboard";
import { useHunkTheme } from "../libraries/theme";
import { createAnalysisClient, type Phase } from "../libraries/progress";
import { buildTree, DocNav, firstDocId, langId, splitId, visibleRows } from "./doc-nav";
import { HelpModal } from "./help-modal";
import { NotePanel, PairView } from "./pair-view";
import { PromptBar } from "./prompt-bar";
import { ThemeModal } from "./theme-modal";
import { useToast } from "./toaster";

const NAV_WIDTH = 34;
const NAV_GAP = 2;

interface DocCache {
  combined: Combined | null;
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
      return `writing note${pct.length > 0 ?pct:" · prefilling.."}`;
    default:
      return "working…";
  }
}

export function ReviewScreen(_props: { initialFile?: unknown }) {
  const { width } = useTerminalDimensions();
  const { name, theme } = useHunkTheme();
  const renderer = useRenderer();
  const toast = useToast();

  useCopyOnSelect((copied) =>
    toast(copied ? "copied to clipboard" : "could not copy to clipboard", {
      variant: copied ? "info" : "warning",
      duration: 2000,
    }),
  );

  const client = useMemo(() => createAnalysisClient(), []);
  const tree = useMemo(() => buildTree(), []);

  const [selectedId, setSelectedId] = useState(() => firstDocId(tree));
  const [modal, setModal] = useState<ModalKind | null>(null);
  const [showHighlights, setShowHighlights] = useState(true);
  const [showGold, setShowGold] = useState(true);
  const [status, setStatus] = useState("idle");
  const [phase, setPhase] = useState<Phase | null>(null);
  const [pass, setPass] = useState<number | undefined>(undefined);
  const [progress, setProgress] = useState(0);
  const [head, setHead] = useState<Head | null>(null);
  const [combined, setCombined] = useState<Combined | null>(null);
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
  const runningRef = useRef(false);
  selectedRef.current = selectedId;

  const doc = findDoc(selectedId);
  const mainWidth = Math.max(40, width - NAV_WIDTH - NAV_GAP);

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
    runningRef.current = false;
    const c = cache.current[selectedId];
    setCombined(c?.combined ?? null);
    setNote(c?.note ?? "");
    setSignal(c?.signal ?? null);
    setStatus("idle");
    setPhase(null);
    setPass(undefined);
    setProgress(0);
    setRunning(false);
    // Any document change lands on the ground-truth highlighted view.
    setShowGold(true);
    setShowHighlights(true);
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
      setHead(target);
      const cached = cache.current[id];
      if (cached?.combined) {
        setCombined(cached.combined);
        setSignal(cached.signal);
        setNote(cached.note);
        setShowGold(false);
        setStatus("idle");
        return;
      }
      //FIXME: one capture + one agent note per document; only the highlight is per-mode.
      if (runningRef.current) return;
      runningRef.current = true;
      setRunning(true);
      setPhase("loading");
      setPass(undefined);
      setProgress(0);
      setCombined(null);
      setSignal(null);
      setNote("");
      setStatus(`computing layerwise + ${HEAD_LABEL[target]}…`);

      try {
        const result = await client.analyse(doc, {
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
          onScores: (_a, _b, sig, comb) => {
            if (selectedRef.current !== id) return;
            setCombined(comb ?? null);
            setSignal(sig);
            // Model output is in: switch off ground truth so the scores + note show.
            setShowGold(false);
          },
          onNote: (text) => {
            if (selectedRef.current === id) setNote(text);
          },
          onNoteProgress: (fraction) => {
            if (selectedRef.current === id) setProgress(fraction);
          },
        });

        if (selectedRef.current === id) {
          setCombined(result.combined);
          setSignal(result.signal);
          setNote(result.note);
          if (result.combined) {
            cache.current[id] = { combined: result.combined, signal: result.signal, note: result.note };
          }
          setRunning(false);
          setPhase(null);
          setPass(undefined);
          setProgress(0);
          setStatus("idle");
        }
      } finally {
        runningRef.current = false;
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
      else if (key.name === "d") void analyse("combined");
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
    } else if (key.name === "space" || key.name === "return") {
      key.preventDefault();
      openRow();
    }
  });

  const onSubmit = (text: string) => {
    runCommand(text, {
      quit,
      openModal: setModal,
      toast: (message, variant) => toast(message, variant ? { variant } : undefined),
      runGru: () => void analyse("gru"),
      runDiffalign: () => void analyse("diffalign"),
      runCombined: () => void analyse("combined"),
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
              onCursor={setCursorIndex}
            />
          </scrollbox>
        </box>
        <box style={{ width: NAV_GAP, height: "100%", backgroundColor: theme.background }} />
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
                combined={combined}
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
            head={head}
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

      <box
        style={{ width: "100%", height: 1, flexShrink: 0 }}
        flexDirection="row"
        justifyContent="center"
      >
        <text
          fg={theme.muted}
        >{`[↑/↓] doc  ·  [^e] highlights  ·  [^g] ground truth  ·  [^r] ${HEAD_LABEL.gru}  ·  [^k] ${HEAD_LABEL.diffalign}  ·  [^d] ${HEAD_LABEL.combined}  ·  [^p] help  ·  [^t] theme  ·  [^c] quit`}</text>
      </box>

      {modal === "help" && <HelpModal onClose={() => setModal(null)} />}
      {modal === "theme" && <ThemeModal onClose={() => setModal(null)} />}
    </box>
  );
}
