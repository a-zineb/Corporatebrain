import { ArrowUp, Paperclip, Sparkles } from 'lucide-react';
import { useRef } from 'react';

export function AskInputBar({
  value,
  onChange,
  onSubmit,
  onAttach,
  placeholder = 'Ask Anything…',
  busy = false,
  disabled = false,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onAttach?: (file: File) => void;
  placeholder?: string;
  busy?: boolean;
  disabled?: boolean;
}) {
  const picker = useRef<HTMLInputElement>(null);
  const canSend = value.trim().length > 0 && !busy && !disabled;

  return (
    <div className="ask-input-bar-shell">
      <div className="ask-input-bar-wrap">
        <form
          className="ask-input-bar"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSend) onSubmit();
        }}
      >
        <div className="ask-input-bar__input-row">
          <Sparkles
            className="ask-input-bar__sparkle"
            size={20}
            aria-hidden="true"
          />
          <textarea
            className="ask-input-bar__field"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
            rows={2}
            aria-label={placeholder}
            disabled={disabled || busy}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (canSend) onSubmit();
              }
            }}
          />
        </div>

        <div className="ask-input-bar__toolbar">
          {onAttach ? (
            <button
              type="button"
              className="ask-input-bar__attach"
              onClick={() => picker.current?.click()}
              disabled={disabled || busy}
            >
              <Paperclip size={16} aria-hidden="true" />
              <span>Attach</span>
            </button>
          ) : (
            <span />
          )}

          <button
            type="submit"
            className="ask-input-bar__send"
            disabled={!canSend}
            aria-label="Send message"
          >
            <ArrowUp size={16} aria-hidden="true" />
          </button>
        </div>

        {onAttach && (
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
        )}
      </form>
      </div>
    </div>
  );
}
