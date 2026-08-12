import {
  Copy,
  Download,
  Pencil,
  RefreshCw,
  ThumbsDown,
  ThumbsUp,
} from 'lucide-react';

export function MessageActions({ text }: { text: string }) {
  function copy() {
    void navigator.clipboard.writeText(text);
  }

  return (
    <div className="message-actions">
      <button onClick={copy} title="Copy" aria-label="Copy">
        <Copy size={14} />
      </button>
      <button title="Good response" aria-label="Good response">
        <ThumbsUp size={14} />
      </button>
      <button title="Bad response" aria-label="Bad response">
        <ThumbsDown size={14} />
      </button>
      <button title="Edit" aria-label="Edit">
        <Pencil size={14} />
      </button>
      <button title="Download" aria-label="Download">
        <Download size={14} />
      </button>
      <button title="Regenerate" aria-label="Regenerate">
        <RefreshCw size={14} />
      </button>
    </div>
  );
}
