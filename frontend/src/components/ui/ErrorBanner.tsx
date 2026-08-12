import { AlertTriangle, RefreshCw, X } from 'lucide-react';

export function ErrorBanner({
  message,
  onRetry,
  onDismiss,
}: {
  message: string;
  onRetry?: () => void;
  onDismiss?: () => void;
}) {
  return (
    <div className="error-banner" role="alert">
      <AlertTriangle size={18} />
      <div className="error-banner__text">
        <strong>Unable to complete this action</strong>
        <p>{message}</p>
      </div>
      {onRetry && (
        <button type="button" className="error-banner__retry" onClick={onRetry}>
          <RefreshCw size={14} />
          Retry
        </button>
      )}
      {onDismiss && (
        <button
          type="button"
          className="error-banner__dismiss"
          onClick={onDismiss}
          aria-label="Dismiss"
        >
          <X size={16} />
        </button>
      )}
    </div>
  );
}
