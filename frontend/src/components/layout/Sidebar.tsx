import {
  ChevronDown,
  ChevronRight,
  FileText,
  History,
  MessageSquare,
  Network,
  Search,
  Settings,
} from 'lucide-react';
import { FormEvent, useEffect, useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { getRecentChats, workspaces } from '../../data/mockData';
import { useProfileSnapshot } from '../../hooks/useUserProfile';
import { ProfileEditModal } from '../ui/ProfileEditModal';
import { useUserProfile } from '../../hooks/useUserProfile';
import type { Theme } from '../../types';

export function Sidebar({
  theme,
  setTheme,
}: {
  theme: Theme;
  setTheme: (t: Theme) => void;
}) {
  const navigate = useNavigate();
  const profile = useProfileSnapshot();
  const { saveProfile } = useUserProfile();
  const [query, setQuery] = useState('');
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [workspace, setWorkspace] = useState<(typeof workspaces)[number]>(workspaces[0]);
  const [recentOpen, setRecentOpen] = useState(true);
  const [recentChats, setRecentChats] = useState(getRecentChats(5));
  const [profileOpen, setProfileOpen] = useState(false);
  const isDark =
    theme === 'dark' ||
    (theme === 'system' &&
      matchMedia('(prefers-color-scheme: dark)').matches);

  useEffect(() => {
    const refresh = () => setRecentChats(getRecentChats(5));
    refresh();
    window.addEventListener('cb:history-updated', refresh);
    return () => window.removeEventListener('cb:history-updated', refresh);
  }, []);

  function searchDocs(event: FormEvent) {
    event.preventDefault();
    if (query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  }

  function toggleTheme() {
    setTheme(isDark ? 'light' : 'dark');
  }

  function restoreChat(chat: { id: string }) {
    const all = JSON.parse(sessionStorage.getItem('cb-history') ?? '[]') as Array<{
      id: string;
      title: string;
      date: string;
      document?: string;
      messages: unknown[];
    }>;
    const full = all.find((c) => c.id === chat.id);
    if (full) {
      sessionStorage.setItem('cb-restore-chat', JSON.stringify(full));
      navigate('/');
    }
  }

  return (
    <>
      <aside className="sidebar glass-card">
        <div className="sidebar__brand">
          <span className="sidebar__logo">Corporate Brain</span>
        </div>

        <div className="workspace-switcher">
          <button
            type="button"
            className="workspace-switcher__btn"
            onClick={() => setWorkspaceOpen((v) => !v)}
          >
            <span>{workspace.icon} {workspace.label}</span>
            <ChevronDown size={14} />
          </button>
          {workspaceOpen && (
            <div className="workspace-switcher__menu surface-card">
              {workspaces.map((ws) => (
                <button
                  key={ws.id}
                  type="button"
                  onClick={() => {
                    setWorkspace(ws);
                    setWorkspaceOpen(false);
                  }}
                >
                  {ws.icon} {ws.label}
                </button>
              ))}
            </div>
          )}
        </div>

        <form className="sidebar__search" onSubmit={searchDocs}>
          <Search size={16} />
          <input
            aria-label="Search documents"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search documents..."
          />
          <kbd>⌘K</kbd>
        </form>

        <nav className="sidebar__nav">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `sidebar__link${isActive ? ' active' : ''}`
            }
            end
          >
            <MessageSquare size={18} />
            AI Chat
          </NavLink>

          {recentChats.length > 0 && (
            <div className="sidebar__recent">
              <button
                type="button"
                className="sidebar__recent-toggle"
                onClick={() => setRecentOpen((v) => !v)}
              >
                {recentOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                Recent
              </button>
              {recentOpen &&
                recentChats.map((chat) => (
                  <button
                    key={chat.id}
                    type="button"
                    className="sidebar__recent-item"
                    onClick={() => restoreChat(chat)}
                    title={chat.title}
                  >
                    {chat.title}
                  </button>
                ))}
            </div>
          )}

          <NavLink
            to="/documents"
            className={({ isActive }) =>
              `sidebar__link${isActive ? ' active' : ''}`
            }
          >
            <FileText size={18} />
            Documents
          </NavLink>
          <NavLink
            to="/search"
            className={({ isActive }) =>
              `sidebar__link${isActive ? ' active' : ''}`
            }
          >
            <Search size={18} />
            Search
          </NavLink>
          <NavLink
            to="/graph"
            className={({ isActive }) =>
              `sidebar__link${isActive ? ' active' : ''}`
            }
          >
            <Network size={18} />
            Graph View
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) =>
              `sidebar__link${isActive ? ' active' : ''}`
            }
          >
            <History size={18} />
            History
          </NavLink>
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `sidebar__link${isActive ? ' active' : ''}`
            }
          >
            <Settings size={18} />
            Settings
          </NavLink>
        </nav>

        <button
          type="button"
          className="sidebar__profile surface-card"
          onClick={() => setProfileOpen(true)}
        >
          {profile.avatarUrl ? (
            <img className="sidebar__avatar-img" src={profile.avatarUrl} alt="" />
          ) : (
            <span className="sidebar__avatar">{profile.initials}</span>
          )}
          <div>
            <strong>{profile.name}</strong>
            <small>{profile.plan}</small>
          </div>
        </button>

        <div className="sidebar__footer">
          <div className="theme-switch">
            <span>{isDark ? 'Dark' : 'Light'}</span>
            <button
              className={`theme-switch__toggle${isDark ? '' : ' on'}`}
              onClick={toggleTheme}
              aria-label="Toggle theme"
              type="button"
            >
              <span className="theme-switch__knob" />
            </button>
          </div>
        </div>
      </aside>

      {profileOpen && (
        <ProfileEditModal
          profile={profile}
          onSave={saveProfile}
          onClose={() => setProfileOpen(false)}
        />
      )}
    </>
  );
}
