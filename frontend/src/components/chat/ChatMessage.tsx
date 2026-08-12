import { Brain, User } from 'lucide-react';
import type { ChatResponse, Source } from '../../types';
import { AnswerContent } from './AnswerContent';
import { MessageActions } from './MessageActions';
import { SourceCard } from './SourceCard';

export type ChatMessageData = {
  role: 'user' | 'assistant';
  text: string;
  response?: ChatResponse;
  error?: boolean;
  retryText?: string;
};

export function ChatMessage({
  message,
  showAvatar,
  groupPosition,
  onOpenSource,
  onRetry,
}: {
  message: ChatMessageData;
  showAvatar: boolean;
  groupPosition: 'single' | 'start' | 'middle' | 'end';
  onOpenSource: (source: Source) => void;
  onRetry?: (text: string) => void;
}) {
  const isUser = message.role === 'user';

  return (
    <div
      className={[
        'message',
        `message--${message.role}`,
        groupPosition !== 'single' ? `message--${groupPosition}` : '',
        message.error ? 'message--error' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {!isUser &&
        (showAvatar ? (
          <div className="message__avatar message__avatar--ai">
            <Brain size={16} />
          </div>
        ) : (
          <div className="message__avatar-spacer" aria-hidden="true" />
        ))}

      <div className="message__content">
        <div
          className={`message__bubble${isUser ? ' message__bubble--user' : ' message__bubble--assistant'}`}
        >
          {message.error ? (
            <div className="message__error">
              <p>{message.text}</p>
              {message.retryText && onRetry && (
                <button
                  type="button"
                  className="message__retry"
                  onClick={() => onRetry(message.retryText!)}
                >
                  Retry
                </button>
              )}
            </div>
          ) : (
            <AnswerContent response={message.response} text={message.text} />
          )}
        </div>

        {!isUser && !message.error && (
          <MessageActions text={message.text} />
        )}

        {message.response?.sources.length ? (
          <details className="sources-group surface-card">
            <summary>Sources ({message.response.sources.length})</summary>
            {message.response.sources.map((item) => (
              <SourceCard
                key={`${item.file_hash}-${item.block_id}`}
                source={item}
                onOpen={onOpenSource}
              />
            ))}
          </details>
        ) : null}
      </div>

      {isUser &&
        (showAvatar ? (
          <div className="message__avatar message__avatar--user">
            <User size={16} />
          </div>
        ) : (
          <div className="message__avatar-spacer" aria-hidden="true" />
        ))}
    </div>
  );
}

export function ChatTypingIndicator({ label }: { label: string }) {
  return (
    <div className="message message--assistant message--typing">
      <div className="message__avatar message__avatar--ai">
        <Brain size={16} />
      </div>
      <div className="message__content">
        <div className="message__bubble message__bubble--assistant message__bubble--typing">
          <span className="typing-dots" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <span className="typing-label">{label}</span>
        </div>
      </div>
    </div>
  );
}

export function getGroupPosition(
  messages: ChatMessageData[],
  index: number,
): 'single' | 'start' | 'middle' | 'end' {
  const current = messages[index];
  const prev = messages[index - 1];
  const next = messages[index + 1];
  const sameAsPrev = prev?.role === current.role;
  const sameAsNext = next?.role === current.role;

  if (!sameAsPrev && !sameAsNext) return 'single';
  if (!sameAsPrev && sameAsNext) return 'start';
  if (sameAsPrev && sameAsNext) return 'middle';
  return 'end';
}

export function shouldShowAvatar(
  messages: ChatMessageData[],
  index: number,
): boolean {
  const current = messages[index];
  const next = messages[index + 1];
  return !next || next.role !== current.role;
}
