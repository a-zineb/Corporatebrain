import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/corporateBrain';
import type { DocumentItem } from '../types';
import { DocumentGraph } from '../components/graph/DocumentGraph';
import { ErrorBanner } from '../components/ui/ErrorBanner';
import { SurfaceCard } from '../components/ui/PageShell';
import { SkeletonList } from '../components/ui/Skeleton';
import { sampleDocuments } from '../data/mockData';
import { useToast } from '../hooks/useToast';

export function GraphPage() {
  const { showError } = useToast();
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [mockSamples, setMockSamples] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [bannerError, setBannerError] = useState('');

  const allDocs = [...docs, ...mockSamples];

  function load() {
    setLoading(true);
    setBannerError('');
    return api
      .documents()
      .then(setDocs)
      .catch((e) => {
        const msg = e instanceof Error ? e.message : 'Could not load documents.';
        setBannerError(msg);
        showError(msg, load);
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    void load();
    try {
      const samples = sessionStorage.getItem('cb-mock-samples');
      if (samples) setMockSamples(JSON.parse(samples) as DocumentItem[]);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    const onSamples = () => {
      try {
        const samples = sessionStorage.getItem('cb-mock-samples');
        if (samples) setMockSamples(JSON.parse(samples) as DocumentItem[]);
      } catch {
        /* ignore */
      }
    };
    window.addEventListener('cb:samples-updated', onSamples);
    return () => window.removeEventListener('cb:samples-updated', onSamples);
  }, []);

  return (
    <section className="page">
      <SurfaceCard className="page-card">
        <header className="page-card__header">
          <div>
            <span className="eyebrow">Corpus Visualization</span>
            <h1>Graph View</h1>
            <p>
              Obsidian-style map of document connections based on shared keywords
              (5+ matches).
            </p>
          </div>
          {allDocs.length < 3 && (
            <button
              type="button"
              className="button"
              onClick={() => {
                setMockSamples(sampleDocuments);
                sessionStorage.setItem(
                  'cb-mock-samples',
                  JSON.stringify(sampleDocuments),
                );
                window.dispatchEvent(new CustomEvent('cb:samples-updated'));
              }}
            >
              Load sample docs for demo
            </button>
          )}
        </header>

        {bannerError && (
          <ErrorBanner message={bannerError} onRetry={load} onDismiss={() => setBannerError('')} />
        )}

        {loading ? (
          <SkeletonList count={3} />
        ) : (
          <DocumentGraph documents={allDocs} />
        )}

        <p className="graph-footer">
          Also available from{' '}
          <Link to="/search">Search</Link> via &quot;View as Graph&quot;.
        </p>
      </SurfaceCard>
    </section>
  );
}
