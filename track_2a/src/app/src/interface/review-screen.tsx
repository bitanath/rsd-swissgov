// TODO the entire review screen
import { useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/react";
import { useMemo, useState } from "react";

import type { AutocompleteOption } from "../components/autocomplete";
import { Alert } from "../components/alert";
import { useCopyOnSelect } from "../libraries/clipboard";
import { COMMANDS, runCommand, type LayoutMode, type ModalKind } from "../libraries/commands";
import { dummyDiff, dummyFiles } from "../libraries/fixtures";
import type { HunkDiffFileInput } from "../libraries/hunk/opentui";
import { HunkDiffBody } from "../libraries/hunk/opentui/HunkDiffBody";
import { HunkFileNav } from "../libraries/hunk/opentui/HunkFileNav";
import { useHunkTheme } from "../libraries/theme";
import { HelpModal } from "./help-modal";
import { PromptBar } from "./prompt-bar";
import { ThemeModal } from "./theme-modal";
import { useToast } from "./toaster";

const NAV_WIDTH = 34;

const COMMAND_OPTIONS: AutocompleteOption[] = COMMANDS.map((command) => ({
  name: command.name,
  label: command.label,
  description: command.description,
}));

export function ReviewScreen({ initialFile }: { initialFile?: HunkDiffFileInput }) {
  const { width } = useTerminalDimensions();
  const { name, theme } = useHunkTheme();
  const renderer = useRenderer();
  const toast = useToast();

  const files = useMemo(() => (initialFile ? [initialFile] : dummyFiles), [initialFile]);
  const activeFile = files[0] ?? dummyDiff;

  const [selectedFileId, setSelectedFileId] = useState(activeFile.id);
  const [layout, setLayout] = useState<LayoutMode>("split");
  const [modal, setModal] = useState<ModalKind | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const file = files.find((candidate) => candidate.id === selectedFileId) ?? activeFile;
  const diffWidth = Math.max(24, width - NAV_WIDTH);

  const quit = () => {
    renderer.destroy();
    process.exit(0);
  };

  useCopyOnSelect((copied) =>
    toast(copied ? "Copied to clipboard" : "Copy failed", { variant: copied ? "success" : "error" }),
  );

  useKeyboard((key) => {
    if (!key.ctrl) return;
    if (key.name === "c") quit();
    else if (key.name === "t") setModal("theme");
    else if (key.name === "r") setRefreshKey((current) => current + 1);
  });

  const onSubmit = (text: string) => {
    runCommand(text, {
      layout,
      setLayout,
      quit,
      openModal: setModal,
      toast: (message, variant) => toast(message, variant ? { variant } : undefined),
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
        <Alert variant="info" title="aperdiff" message={`${name} · ${layout}`} />
      </box>

      <box
        style={{
          flexGrow: 1,
          flexDirection: "row",
          minHeight: 0,
          backgroundColor: theme.background,
        }}
      >
        <box style={{ width: NAV_WIDTH, height: "100%", backgroundColor: theme.background }}>
          <scrollbox width="100%" height="100%" scrollY>
            <HunkFileNav
              files={files}
              selectedFileId={selectedFileId}
              width={NAV_WIDTH}
              theme={name}
              onSelectFile={setSelectedFileId}
            />
          </scrollbox>
        </box>

        <box style={{ width: diffWidth, height: "100%", backgroundColor: theme.background }}>
          <scrollbox
            key={`${file.id}:${layout}:${refreshKey}`}
            width="100%"
            height="100%"
            scrollY
            viewportCulling
          >
            <HunkDiffBody file={file} width={diffWidth} theme={name} canonicalLayout={layout} />
          </scrollbox>
        </box>
      </box>

      <PromptBar options={COMMAND_OPTIONS} onSubmit={onSubmit} focused={modal === null} />

      <box style={{ width: "100%", height: 1, flexDirection: "row", flexShrink: 0 }}>
        <text
          fg={theme.muted}
        >{`[^t] theme  ·  [^r] refresh  ·  [^c] quit  ·  drag to copy`}</text>
      </box>

      {modal === "help" && <HelpModal onClose={() => setModal(null)} />}
      {modal === "theme" && <ThemeModal onClose={() => setModal(null)} />}
    </box>
  );
}
