import { Paperclip, Send } from 'lucide-react';
import { useRef, useState } from 'react';

export function ChatComposer({
  onSend,
  onAttach,
  busy,
  uploading,
}: {
  onSend: (text: string) => void;
  onAttach: (file: File) => void;
  busy: boolean;
  uploading: boolean;
}) {
  const [text, setText] = useState('');
  const picker = useRef<HTMLInputElement>(null);

  return (
    <div className="composer-shell">
      <div className="composer-border-wrap">
        <form
          className="composer composer-inner"
          onSubmit={(e) => {
            e.preventDefault();
            if (text.trim() && !busy) {
              onSend(text.trim());
              setText('');
            }
          }}
        >
          <button
            type="button"
            className="composer__attach"
            onClick={() => picker.current?.click()}
            disabled={uploading}
            aria-label="Attach a document"
          >
            <Paperclip size={18} />
          </button>

          <input
            ref={picker}
            hidden
            type="file"
            accept=".pdf,.docx,.doc,.xlsx,.csv,.zip"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onAttach(file);
              e.currentTarget.value = '';
            }}
          />

          <textarea
            aria-label="Ask Corporate Brain"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Ask Corporate Brain anything about your documents..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                e.currentTarget.form?.requestSubmit();
              }
            }}
          />

          <button
            className="composer__send"
            disabled={busy || !text.trim()}
            aria-label="Send"
            type="submit"
          >
            <Send size={18} />
          </button>
        </form>
      </div>

      {uploading && (
        <p className="composer__status">Preparing document…</p>
      )}
    </div>
  );
}
