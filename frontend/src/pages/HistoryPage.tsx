import { Clock, FileText, MessageSquare } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { SurfaceCard } from '../components/ui/PageShell';
import { computeHistoryMetrics } from '../data/mockData';

interface SavedChat {
  id: string;
  title: string;
  date: string;
  document?: string;
  messages: unknown[];
  pinned?: boolean;
  archived?: boolean;
}

type GroupTab = 'all' | 'pinned' | 'archived';

export function HistoryPage() {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [tab, setTab] = useState<GroupTab>('all');
  const chats = JSON.parse(
    sessionStorage.getItem('cb-history') ?? '[]',
  ) as SavedChat[];

  const metrics = useMemo(() => computeHistoryMetrics(chats), [chats]);

  const filtered = chats.filter((chat) => {
    const matchesQuery =
      !query.trim() ||
      chat.title.toLowerCase().includes(query.toLowerCase()) ||
      chat.document?.toLowerCase().includes(query.toLowerCase());
    if (tab === 'pinned') return matchesQuery && chat.pinned;
    if (tab === 'archived') return matchesQuery && chat.archived;
    return matchesQuery && !chat.archived;
  });

  function restore(chat: SavedChat) {
    sessionStorage.setItem('cb-restore-chat', JSON.stringify(chat));
    navigate('/');
  }

  function startNewChat() {
    window.dispatchEvent(new CustomEvent('cb:new-chat'));
    navigate('/');
  }

  return (
    <section className="page">
      <SurfaceCard className="page-card">
        <header>
          <h1>History</h1>
          <p>Recent conversations from this browser session.</p>
        </header>

        <div className="metrics-row">
          <SurfaceCard className="metric-card">
            <strong>{metrics.totalSessions}</strong>
            <span>Total Sessions</span>
          </SurfaceCard>
          <SurfaceCard className="metric-card">
            <strong>{metrics.questionsAsked}</strong>
            <span>Questions Asked</span>
          </SurfaceCard>
          <SurfaceCard className="metric-card">
            <strong>
              {metrics.avgSessionMinutes > 0
                ? `${metrics.avgSessionMinutes} min`
                : '—'}
            </strong>
            <span>Avg. Session Length</span>
          </SurfaceCard>
        </div>

        <div className="history-toolbar">
          <input
            className="search-input"
            placeholder="Search history…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="filter-tabs">
            {(['all', 'pinned', 'archived'] as GroupTab[]).map((t) => (
              <button
                key={t}
                type="button"
                className={tab === t ? 'active' : ''}
                onClick={() => setTab(t)}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {filtered.length ? (
          <div className="history-list">
            {filtered.map((chat) => (
              <button
                onClick={() => restore(chat)}
                key={chat.id}
                className="history-card surface-card"
                type="button"
              >
                <Clock />
                <div>
                  <strong>{chat.title}</strong>
                  <small>
                    {new Date(chat.date).toLocaleString()} ·{' '}
                    {chat.messages.length} messages
                  </small>
                  {chat.document && (
                    <span>
                      <FileText size={13} /> {chat.document}
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="empty-state surface-card">
            <MessageSquare size={32} />
            <h2>No conversations yet</h2>
            <p>
              Your chat logs with Corporate Brain will automatically save here.
            </p>
            <button type="button" className="button" onClick={startNewChat}>
              💬 Start a New Chat
            </button>
          </div>
        )}
      </SurfaceCard>
    </section>
  );
}
