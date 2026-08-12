import { useEffect, useState } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import type { Theme } from './types';
import { AppLayout } from './components/layout/AppLayout';
import { ChatPage } from './pages/ChatPage';
import { DocumentsPage } from './pages/DocumentsPage';
import { SearchPage } from './pages/SearchPage';
import { SettingsPage } from './pages/SettingsPage';
import { HistoryPage } from './pages/HistoryPage';
import { GraphPage } from './pages/GraphPage';
import { ToastProvider } from './hooks/useToast';

export default function App() {
  const [theme, setThemeState] = useState<Theme>(
    (localStorage.getItem('cb-theme') as Theme) || 'light',
  );

  const setTheme = (value: Theme) => {
    setThemeState(value);
    localStorage.setItem('cb-theme', value);
  };

  useEffect(() => {
    const resolved =
      theme === 'system'
        ? matchMedia('(prefers-color-scheme: dark)').matches
          ? 'dark'
          : 'light'
        : theme;
    document.documentElement.dataset.theme = resolved;
  }, [theme]);

  return (
    <ToastProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout theme={theme} setTheme={setTheme} />}>
            <Route path="/" element={<ChatPage />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/graph" element={<GraphPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route
              path="/settings"
              element={<SettingsPage theme={theme} setTheme={setTheme} />}
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  );
}
