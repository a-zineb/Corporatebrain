import { Command } from 'lucide-react';
import { SystemStatusBadge } from '../ui/SystemStatusBadge';
import { CommandPalette, useCommandPalette } from '../ui/CommandPalette';
import type { ConnectionState } from '../../hooks/useHealth';

export function TopBar({ connectionState }: { connectionState: ConnectionState }) {
  const { open, setOpen } = useCommandPalette();

  return (
    <>
      <header className="top-bar surface-card">
        <button
          type="button"
          className="top-bar__cmd"
          onClick={() => setOpen(true)}
        >
          <Command size={16} />
          <span>Search or jump to…</span>
          <kbd>⌘K</kbd>
        </button>
        <SystemStatusBadge state={connectionState} />
      </header>
      <CommandPalette open={open} onClose={() => setOpen(false)} />
    </>
  );
}
