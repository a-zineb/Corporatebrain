import { useEffect, useState } from 'react';
import { api } from '../api/corporateBrain';
import type { DocumentItem, HealthStatus, Theme } from '../types';
import { ErrorBanner } from '../components/ui/ErrorBanner';
import { SurfaceCard } from '../components/ui/PageShell';
import { useToast } from '../hooks/useToast';

export function SettingsPage({
  theme,
  setTheme,
}: {
  theme: Theme;
  setTheme: (t: Theme) => void;
}) {
  const { showError, showSuccess } = useToast();
  const [health, setHealth] = useState<HealthStatus | null>(
    null,
  );
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [bannerError, setBannerError] = useState('');
  const [chunkSize, setChunkSize] = useState(512);
  const [topK, setTopK] = useState(8);
  const [temperature, setTemperature] = useState(0.3);

  const load = () => {
    setBannerError('');
    void api
      .health()
      .then(setHealth)
      .catch((e) => {
        const msg = e instanceof Error ? e.message : 'Health check failed.';
        setBannerError(msg);
        showError(msg, load);
      });
    void api.documents().then(setDocs).catch(() => {});
  };

  useEffect(() => {
    load();
  }, []);

  const totalChunks = docs.reduce((sum, d) => sum + d.blocks, 0);
  const readyDocs = docs.filter((d) => d.status === 'ready').length;

  function dangerAction(label: string) {
    if (
      confirm(
        `${label} — this is a demo action. No backend changes will be made. Continue?`,
      )
    ) {
      showSuccess('Action queued (demo)', label);
    }
  }

  return (
    <section className="page">
      <SurfaceCard className="page-card">
        <header>
          <h1>Settings</h1>
          <p>System status, AI configuration, and document corpus info.</p>
        </header>

        {bannerError && (
          <ErrorBanner
            message={bannerError}
            onRetry={load}
            onDismiss={() => setBannerError('')}
          />
        )}

        <div className="settings-grid">
          <SurfaceCard className="settings-card">
            <h2>Appearance</h2>
            <label>
              Theme
              <select
                value={theme}
                onChange={(e) => setTheme(e.target.value as Theme)}
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="system">System</option>
              </select>
            </label>
          </SurfaceCard>

          <SurfaceCard className="settings-card">
            <h2>Backend Status</h2>
            <dl className="settings-stats">
              <div>
                <dt>API Health</dt>
                <dd className={health?.status === 'ok' ? 'status-ok' : ''}>
                  {health?.status ?? 'Checking…'}
                </dd>
              </div>
              <div>
                <dt>Service</dt>
                <dd>{health?.service ?? '—'}</dd>
              </div>
            </dl>
          </SurfaceCard>

          <SurfaceCard className="settings-card">
            <h2>Document Corpus</h2>
            <dl className="settings-stats">
              <div>
                <dt>Uploaded files</dt>
                <dd>{docs.length}</dd>
              </div>
              <div>
                <dt>Ready documents</dt>
                <dd>{readyDocs}</dd>
              </div>
              <div>
                <dt>Total chunks</dt>
                <dd>{totalChunks}</dd>
              </div>
            </dl>
          </SurfaceCard>

          <SurfaceCard className="settings-card settings-card--wide">
            <h2>LLM &amp; RAG Pipeline</h2>
            <div className="settings-controls">
              <dl className="settings-stats">
                <div><dt>AI Answer</dt><dd>API-backed</dd></div>
                <div><dt>AI provider configured</dt><dd>{health?.ai_provider_configured ? 'Yes' : 'No'}</dd></div>
              </dl>
              <label>
                Chunk Size ({chunkSize} tokens)
                <input
                  type="range"
                  min={128}
                  max={1024}
                  step={64}
                  value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                />
              </label>
              <label>
                Top-K Retrieval ({topK})
                <input
                  type="range"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
              </label>
              <label>
                Temperature ({temperature.toFixed(1)})
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                />
              </label>
            </div>
          </SurfaceCard>

          {docs.length > 0 && (
            <SurfaceCard className="settings-card settings-card--wide">
              <h2>Document Details</h2>
              <div className="settings-doc-list">
                {docs.map((doc) => (
                  <div key={doc.id} className="settings-doc-row">
                    <strong>{doc.name}</strong>
                    <span>{doc.type.toUpperCase()}</span>
                    <span>{doc.blocks} chunks</span>
                    <span className="doc-status">{doc.status.replaceAll('_', ' ')}</span>
                  </div>
                ))}
              </div>
            </SurfaceCard>
          )}

          <SurfaceCard className="settings-card settings-card--wide danger-zone">
            <h2>Danger Zone</h2>
            <p>Destructive maintenance actions — demo mode only.</p>
            <div className="danger-zone__actions">
              <button
                type="button"
                onClick={() => dangerAction('Re-index All Documents')}
              >
                🔄 Re-index All Documents
              </button>
              <button
                type="button"
                className="danger-zone__destructive"
                onClick={() => dangerAction('Clear Vector Database')}
              >
                🗑️ Clear Vector Database
              </button>
            </div>
          </SurfaceCard>
        </div>
      </SurfaceCard>
    </section>
  );
}
