// CodeCell.jsx (updated with Monaco Editor for persistent syntax highlighting)
import React, { useState, useRef } from 'react';
import { Play, Loader2, Copy, Check, RefreshCw, ClipboardPaste } from 'lucide-react';
import Editor, { loader } from '@monaco-editor/react';
import { useTheme } from '../context/ThemeContext';

// Configure Monaco to use local workers from node_modules
loader.config({
    paths: {
        vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs'
    }
});

export default function CodeCell({ content, handleContentChange, handleRun, isRunning, isCompleted, activeTemplate, onToggleTemplate }) {
    const [copied, setCopied] = useState(false);
    const [pasted, setPasted] = useState(false);
    const editorRef = useRef(null);
    const { isDarkMode } = useTheme();

    const handleCopy = async () => {
        try {
            // Determine what to copy: selected text or all content
            let textToCopy = content;
            if (editorRef.current) {
                const selection = editorRef.current.getSelection();
                const selectedText = editorRef.current.getModel()?.getValueInRange(selection);
                if (selectedText && selectedText.length > 0) {
                    textToCopy = selectedText;
                }
            }

            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(textToCopy);
            } else {
                const textArea = document.createElement('textarea');
                textArea.value = textToCopy;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                try {
                    document.execCommand('copy');
                } finally {
                    document.body.removeChild(textArea);
                }
            }

            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
            alert('Nu s-a putut copia textul. Vă rugăm selectați manual și copiați cu Ctrl+C.');
        }
    };

    const handlePaste = async () => {
        try {
            let clipboardText = '';
            if (navigator.clipboard && navigator.clipboard.readText) {
                clipboardText = await navigator.clipboard.readText();
            } else {
                alert('Clipboard access not supported. Please use Ctrl+V inside the editor.');
                return;
            }
            // Replace ALL content with clipboard text
            handleContentChange(clipboardText);
            setPasted(true);
            setTimeout(() => setPasted(false), 2000);
        } catch (err) {
            console.error('Failed to paste:', err);
            alert('Nu s-a putut face paste. Folosiți Ctrl+V în editor.');
        }
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 overflow-hidden flex flex-col h-full">
            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
                <div className="flex items-center gap-2">
                    {/* Template Toggle Button */}
                    <button
                        onClick={onToggleTemplate}
                        disabled={isCompleted || isRunning}
                        className={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded transition-colors ${activeTemplate === 'tensorflow'
                            ? 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-900/60'
                            : 'bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300 hover:bg-orange-200 dark:hover:bg-orange-900/60'
                            } disabled:opacity-50 disabled:cursor-not-allowed`}
                        title={`Switch to ${activeTemplate === 'tensorflow' ? 'PyTorch' : 'TensorFlow'} template`}
                    >
                        <RefreshCw className="w-3.5 h-3.5" />
                        <span>{activeTemplate === 'tensorflow' ? '🔥 PyTorch' : '🔶 TensorFlow'}</span>
                    </button>
                </div>
                <div className="flex items-center gap-2">
                    {/* Copy Button */}
                    <button
                        onClick={handleCopy}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-gray-700 dark:text-gray-300 text-sm rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                        title="Copy selected text, or all if nothing selected"
                    >
                        {copied ? (
                            <>
                                <Check className="w-4 h-4 text-green-600 dark:text-green-400" />
                                <span className="text-green-600 dark:text-green-400">Copied!</span>
                            </>
                        ) : (
                            <>
                                <Copy className="w-4 h-4" />
                                <span>Copy</span>
                            </>
                        )}
                    </button>

                    {/* Paste Button */}
                    <button
                        onClick={handlePaste}
                        disabled={isCompleted || isRunning}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-gray-700 dark:text-gray-300 text-sm rounded hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        title="Paste clipboard content (replaces all text)"
                    >
                        {pasted ? (
                            <>
                                <Check className="w-4 h-4 text-green-600 dark:text-green-400" />
                                <span className="text-green-600 dark:text-green-400">Pasted!</span>
                            </>
                        ) : (
                            <>
                                <ClipboardPaste className="w-4 h-4" />
                                <span>Paste</span>
                            </>
                        )}
                    </button>

                    {/* Run Button */}
                    <button
                        onClick={handleRun}
                        disabled={isRunning || isCompleted}
                        className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 dark:bg-blue-700 text-white text-sm rounded hover:bg-blue-700 dark:hover:bg-blue-600 disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
                    >
                        {isCompleted ? (
                            'Simulation Completed - No Run'
                        ) : isRunning ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Running...
                            </>
                        ) : (
                            <>
                                <Play className="w-4 h-4" />
                                Run
                            </>
                        )}
                    </button>
                </div>
            </div>
            <div className="flex-1 overflow-hidden">
                <Editor
                    height="100%"
                    defaultLanguage="python"
                    language="python"
                    value={content}
                    onChange={(value) => handleContentChange(value || '')}
                    onMount={(editor) => { editorRef.current = editor; }}
                    theme={isDarkMode ? 'vs-dark' : 'light'}
                    options={{
                        minimap: { enabled: false },
                        fontSize: 14,
                        lineNumbers: 'on',
                        scrollBeyondLastLine: false,
                        automaticLayout: true,
                        tabSize: 4,
                        wordWrap: 'on',
                        readOnly: isCompleted,
                        scrollbar: {
                            vertical: 'auto',
                            horizontal: 'auto',
                        },
                        padding: { top: 10, bottom: 10 },
                        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                    }}
                    loading={
                        <div className="flex items-center justify-center h-full bg-white dark:bg-gray-800">
                            <Loader2 className="w-6 h-6 animate-spin text-blue-600 dark:text-blue-400" />
                        </div>
                    }
                />
            </div>
        </div>
    );
}