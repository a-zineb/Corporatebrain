import { Search } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  commandPaletteItems,
} from '../../data/mockData';

type PaletteItem = {
  id: string;
  label: string;
  group: string;
  path?: string;
  action?: 'new-chat';
  subtitle?: string;
};

function fuzzyMatch(query: string, text: string) {
  const q = query.toLowerCase().trim();
  if (!q) return true;
  return text.toLowerCase().includes(q);
}

export function CommandPalette({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');

  const items = useMemo(() => {
    return commandPaletteItems.filter(item=>item.path!=='/search') as PaletteItem[];
  }, [open]);

  const filtered = items.filter(
    (item) =>
      fuzzyMatch(query, item.label) ||
      fuzzyMatch(query, item.group) ||
      (item.subtitle && fuzzyMatch(query, item.subtitle)),
  );

  useEffect(() => {
    if (!open) setQuery('');
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  function select(item: PaletteItem) {
    if (item.action === 'new-chat') {
      window.dispatchEvent(new CustomEvent('cb:new-chat'));
      navigate('/');
    } else if (item.path) {
      navigate(item.path);
    }
    onClose();
  }

  const groups = [...new Set(filtered.map((i) => i.group))];

  return (
    <div className="cmd-overlay" onClick={onClose} role="presentation">
      <div
        className="cmd-palette surface-card"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Command palette"
      >
        <div className="cmd-palette__search">
          <Search size={18} />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search chats, documents, actions…"
            aria-label="Command search"
          />
          <kbd>Esc</kbd>
        </div>
        <div className="cmd-palette__results">
          {filtered.length === 0 && (
            <p className="cmd-palette__empty">No matches found.</p>
          )}
          {groups.map((group) => (
            <section key={group}>
              <h3>{group}</h3>
              {filtered
                .filter((i) => i.group === group)
                .map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className="cmd-palette__item"
                    onClick={() => select(item)}
                  >
                    <span>{item.label}</span>
                    {item.subtitle && (
                      <small>{item.subtitle}</small>
                    )}
                  </button>
                ))}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

export function useCommandPalette() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen((v) => !v);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return { open, setOpen };
}
