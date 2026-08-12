import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { PageShell } from '../ui/PageShell';
import { useHealth } from '../../hooks/useHealth';
import type { Theme } from '../../types';

export function AppLayout({
  theme,
  setTheme,
}: {
  theme: Theme;
  setTheme: (t: Theme) => void;
}) {
  const { state } = useHealth();

  return (
    <div className="app-shell">
      <Sidebar theme={theme} setTheme={setTheme} />
      <div className="app-main">
        <TopBar connectionState={state} />
        <PageShell>
          <main className="app-content">
            <Outlet />
          </main>
        </PageShell>
      </div>
    </div>
  );
}
