import {
  FileText,
  History,
  LayoutGrid,
  MessageSquarePlus,
  Settings,
} from 'lucide-react';
import { NavLink, useNavigate } from 'react-router-dom';

export function IconRail() {
  const navigate = useNavigate();

  function newChat() {
    window.dispatchEvent(new CustomEvent('cb:new-chat'));
    navigate('/');
  }

  return (
    <aside className="icon-rail glass">
      <div className="icon-rail__logo">
        <span className="logo-mark">CB</span>
      </div>

      <nav className="icon-rail__nav">
        <button
          className="icon-rail__btn"
          onClick={newChat}
          title="New Chat"
          aria-label="New Chat"
        >
          <MessageSquarePlus size={20} />
        </button>
        <NavLink
          to="/"
          className={({ isActive }) =>
            `icon-rail__btn${isActive ? ' active' : ''}`
          }
          title="AI Chat"
          end
        >
          <LayoutGrid size={20} />
        </NavLink>
        <NavLink
          to="/documents"
          className={({ isActive }) =>
            `icon-rail__btn${isActive ? ' active' : ''}`
          }
          title="Documents"
        >
          <FileText size={20} />
        </NavLink>
        <NavLink
          to="/history"
          className={({ isActive }) =>
            `icon-rail__btn${isActive ? ' active' : ''}`
          }
          title="History"
        >
          <History size={20} />
        </NavLink>
      </nav>

      <div className="icon-rail__bottom">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `icon-rail__btn${isActive ? ' active' : ''}`
          }
          title="Settings"
        >
          <Settings size={20} />
        </NavLink>
      </div>
    </aside>
  );
}
