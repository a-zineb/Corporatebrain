import { Bell, MessageCircle, Moon, Sun, User } from 'lucide-react';
import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';

export function TopNav() {
  const [isDark, setIsDark] = useState(
    () => document.documentElement.dataset.theme !== 'light',
  );

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.dataset.theme !== 'light');
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });
    return () => observer.disconnect();
  }, []);

  function toggleTheme() {
    const next = isDark ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('cb-theme', next);
    setIsDark(next === 'dark');
  }

  return (
    <header className="top-nav glass">
      <nav className="top-nav__links">
        <NavLink
          to="/"
          className={({ isActive }) =>
            `top-nav__link${isActive ? ' active' : ''}`
          }
          end
        >
          Dashboard
        </NavLink>
        <NavLink
          to="/"
          className={({ isActive }) =>
            `top-nav__link top-nav__link--chat${isActive ? ' active' : ''}`
          }
          end
        >
          AI Chat
        </NavLink>
        <NavLink
          to="/search"
          className={({ isActive }) =>
            `top-nav__link${isActive ? ' active' : ''}`
          }
        >
          Help
        </NavLink>
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `top-nav__link${isActive ? ' active' : ''}`
          }
        >
          Account
        </NavLink>
      </nav>

      <div className="top-nav__actions">
        <button
          className="top-nav__icon-btn"
          onClick={toggleTheme}
          aria-label="Toggle theme"
        >
          {isDark ? <Sun size={18} /> : <Moon size={18} />}
        </button>
        <button className="top-nav__icon-btn" aria-label="Messages">
          <MessageCircle size={18} />
        </button>
        <button className="top-nav__icon-btn" aria-label="Notifications">
          <Bell size={18} />
        </button>
        <button className="top-nav__avatar" aria-label="Profile">
          <User size={16} />
        </button>
      </div>
    </header>
  );
}
