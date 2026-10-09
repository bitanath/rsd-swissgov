// Inspired by termcn `opentui/input`
import { type KeyEvent, type TextareaRenderable } from "@opentui/core";
import { useEffect, useMemo, useRef, useState } from "react";

import { Autocomplete, type AutocompleteOption } from "../components/autocomplete";
import { useTheme } from "../theme/use-theme";

export function PromptBar({
  options,
  onSubmit,
  focused = true,
  placeholder,
}: {
  options: AutocompleteOption[];
  onSubmit: (text: string) => void;
  focused?: boolean;
  placeholder?: string;
}) {
  const theme = useTheme();
  const inputRef = useRef<TextareaRenderable | null>(null);
  const [inputText, setInputText] = useState("");
  const [acSelected, setAcSelected] = useState(0);
  const [acDismissed, setAcDismissed] = useState(false);

  const acQuery = /^\/(\S*)$/.exec(inputText)?.[1] ?? null;
  const acMatches = useMemo(() => {
    if (acQuery === null) return [];
    const query = acQuery.toLowerCase();
    return options.filter((option) => option.name.startsWith(query));
  }, [acQuery, options]);
  const acOpen = focused && !acDismissed && acQuery !== null && acMatches.length > 0;

  useEffect(() => {
    setAcSelected(0);
  }, [acQuery]);

  const setBuffer = (text: string) => {
    const buffer = inputRef.current?.editBuffer;
    if (buffer) {
      buffer.setText(text);
      buffer.setCursorByOffset(text.length);
    }
    setInputText(text);
  };

  const clear = () => {
    inputRef.current?.editBuffer.setText("");
    setInputText("");
    setAcDismissed(false);
  };

  const submit = () => {
    const text = inputRef.current?.editBuffer.getText() ?? inputText;
    if (!text.trim()) {
      return;
    }
    onSubmit(text);
    clear();
  };

  const handleKey = (key: KeyEvent) => {
    if (!acOpen) return;
    const count = acMatches.length;
    if (key.name === "up" || key.name === "down") {
      key.preventDefault();
      setAcSelected((current) => (key.name === "up" ? (current - 1 + count) % count : (current + 1) % count));
      return;
    }
    if (key.name === "return") {
      const option = acMatches[acSelected] ?? acMatches[0];
      if (option) {
        key.preventDefault();
        onSubmit(`/${option.name}`);
        clear();
      }
      return;
    }
    if (key.name === "tab") {
      const option = acMatches[acSelected] ?? acMatches[0];
      if (option) {
        key.preventDefault();
        setBuffer(`/${option.name} `);
      }
      return;
    }
    if (key.name === "escape") {
      key.preventDefault();
      setAcDismissed(true);
    }
  };

  return (
    <box style={{ width: "100%", flexDirection: "column", flexShrink: 0 }}>
      {acOpen && <Autocomplete options={acMatches} selected={acSelected} />}
      <box
        style={{
          width: "100%",
          borderStyle: "rounded",
          borderColor: focused ? theme.colors.focusRing : theme.colors.border,
          paddingLeft: 1,
          paddingRight: 1,
        }}
      >
        <textarea
          ref={inputRef}
          focused={focused}
          placeholder={placeholder ?? "Type a /command (e.g. /help)"}
          height={1}
          minHeight={1}
          keyBindings={acOpen ? [] : [{ name: "return", action: "submit" }]}
          onSubmit={submit}
          onKeyDown={handleKey}
          onContentChange={() => {
            setInputText(inputRef.current?.editBuffer.getText() ?? "");
            setAcDismissed(false);
          }}
          textColor={theme.colors.foreground}
          focusedTextColor={theme.colors.foreground}
          backgroundColor="transparent"
          focusedBackgroundColor="transparent"
          placeholderColor={theme.colors.mutedForeground}
          cursorColor={theme.colors.focusRing}
          cursorStyle={{ style: "block", blinking: false }}
        />
      </box>
    </box>
  );
}
