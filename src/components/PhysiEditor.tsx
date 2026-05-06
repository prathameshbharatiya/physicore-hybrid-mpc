import React, { useEffect, useRef } from 'react';
import { EditorView, lineNumbers, highlightActiveLine, keymap } from '@codemirror/view';
import { EditorState, Extension } from '@codemirror/state';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { indentOnInput, syntaxHighlighting, defaultHighlightStyle, bracketMatching } from '@codemirror/language';
import { python } from '@codemirror/lang-python';
import { yaml } from '@codemirror/lang-yaml';
import { oneDark } from '@codemirror/theme-one-dark';

interface PhysiEditorProps {
  code: string;
  language?: 'python' | 'yaml';
  onChange?: (value: string) => void;
  readOnly?: boolean;
}

function buildExtensions(language: 'python' | 'yaml', readOnly: boolean, onChange?: (v: string) => void): Extension[] {
  const lang = language === 'yaml' ? yaml() : python();
  const exts: Extension[] = [
    history(),
    lineNumbers(),
    highlightActiveLine(),
    indentOnInput(),
    bracketMatching(),
    syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
    keymap.of([...defaultKeymap, ...historyKeymap]),
    lang,
    oneDark,
    EditorView.theme({
      '&': { background: 'transparent', height: '100%' },
      '.cm-scroller': { fontFamily: 'monospace', fontSize: '12px', overflowY: 'auto' },
      '.cm-gutters': { background: '#0d0f12', borderRight: '1px solid #1e2229' },
    }),
  ];

  if (readOnly) {
    exts.push(EditorState.readOnly.of(true));
  } else if (onChange) {
    exts.push(EditorView.updateListener.of(update => {
      if (update.docChanged) onChange(update.state.doc.toString());
    }));
  }

  return exts;
}

export default function PhysiEditor({ code, language = 'python', onChange, readOnly = false }: PhysiEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const view = new EditorView({
      state: EditorState.create({
        doc: code,
        extensions: buildExtensions(language, readOnly, onChange),
      }),
      parent: containerRef.current,
    });

    viewRef.current = view;
    return () => { view.destroy(); viewRef.current = null; };
  }, [language, readOnly]);

  // Sync external code changes when readOnly
  useEffect(() => {
    const view = viewRef.current;
    if (!view || !readOnly) return;
    const current = view.state.doc.toString();
    if (current !== code) {
      view.dispatch({ changes: { from: 0, to: current.length, insert: code } });
    }
  }, [code, readOnly]);

  return <div ref={containerRef} className="h-full overflow-auto" />;
}
