import { MessageSquare, Plus } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface SavedChat {
  id: string;
  title: string;
  date: string;
  document?: string;
  messages: unknown[];
}

function groupChats(chats: SavedChat[]) {
  const now = Date.now();
  const groups: { label: string; items: SavedChat[] }[] = [];
  const today: SavedChat[] = [];
  const recent: SavedChat[] = [];
  const older: SavedChat[] = [];

  chats.forEach((chat) => {
    const diff = now - new Date(chat.date).getTime();
    const days = diff / (1000 * 60 * 60 * 24);
    if (days < 1) today.push(chat);
    else if (days < 7) recent.push(chat);
    else older.push(chat);
  });

  if (today.length) groups.push({ label: 'Today', items: today });
  if (recent.length) groups.push({ label: '5 Days Ago', items: recent });
  if (older.length) groups.push({ label: '7 Days Ago', items: older });
  return groups;
}

export function HistorySidebar({ onNewChat }: { onNewChat: () => void }) {
  const navigate = useNavigate();
  const chats = JSON.parse(
    sessionStorage.getItem('cb-history') ?? '[]',
  ) as SavedChat[];
  const groups = groupChats(chats);

  function restore(chat: SavedChat) {
    sessionStorage.setItem('cb-restore-chat', JSON.stringify(chat));
    navigate('/');
  }

  return (
    <aside className="history-sidebar glass">
      <header className="history-sidebar__header">
        <h2>History Chat</h2>
        <button className="history-sidebar__new" onClick={onNewChat}>
          <Plus size={14} />
          New Chat
        </button>
      </header>

      <div className="history-sidebar__list">
        {groups.length === 0 && (
          <p className="history-sidebar__empty">
            No conversations yet. Start chatting to build your history.
          </p>
        )}
        {groups.map((group) => (
          <section key={group.label} className="history-sidebar__group">
            <h3>{group.label}</h3>
            {group.items.map((chat) => (
              <button
                key={chat.id}
                className="history-sidebar__item"
                onClick={() => restore(chat)}
              >
                <MessageSquare size={16} />
                <span>{chat.title}</span>
              </button>
            ))}
          </section>
        ))}
      </div>
    </aside>
  );
}
