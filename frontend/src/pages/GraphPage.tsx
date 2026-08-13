import { useEffect, useState } from 'react';
import { api } from '../api/corporateBrain';
import type { DocumentItem } from '../types';
import { DocumentGraph } from '../components/graph/DocumentGraph';
import { ErrorBanner } from '../components/ui/ErrorBanner';
import { SurfaceCard } from '../components/ui/PageShell';
import { SkeletonList } from '../components/ui/Skeleton';
import { useToast } from '../hooks/useToast';

export function GraphPage() {
  const { showError } = useToast();
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [bannerError, setBannerError] = useState('');


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
        </header>

        {bannerError && (
          <ErrorBanner message={bannerError} onRetry={load} onDismiss={() => setBannerError('')} />
        )}

        {loading ? (
          <SkeletonList count={3} />
        ) : (
          <DocumentGraph documents={docs} />
        )}

        <p className="graph-footer">The graph contains only prepared documents from the real corpus.</p>
      </SurfaceCard>
    </section>
  );
}
