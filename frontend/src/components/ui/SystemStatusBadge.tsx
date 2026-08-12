import type { ConnectionState } from '../../hooks/useHealth';

const labels: Record<ConnectionState, { icon: string; text: string }> = {
  ready: { icon: '🟢', text: 'AI Engine Ready' },
  connecting: { icon: '⚠️', text: 'Connecting…' },
  error: { icon: '🔴', text: 'Offline' },
};

export function SystemStatusBadge({ state }: { state: ConnectionState }) {
  const { icon, text } = labels[state];
  return (
    <span className={`status-badge status-badge--${state}`}>
      <span aria-hidden="true">{icon}</span>
      {text}
    </span>
  );
}
