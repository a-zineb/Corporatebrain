import { useEffect, useMemo, useState } from 'react';
import {
  Grid3x3,
  LayoutList,
  Sparkles,
} from 'lucide-react';
import { api } from '../api/corporateBrain';
import type { DocumentItem } from '../types';
import { DocumentCard } from '../components/documents/DocumentCard';
import { DragDropZone } from '../components/ui/DragDropZone';
import { ErrorBanner } from '../components/ui/ErrorBanner';
import { SurfaceCard } from '../components/ui/PageShell';
import { SkeletonList } from '../components/ui/Skeleton';
import { useToast } from '../hooks/useToast';
import {
  maxFileSizeMb,
  sampleDocuments,
  supportedFormats,
} from '../data/mockData';

import {
  countByFileType,
  fileTypeTabs,
  filterByFileType,
  type FileTypeCategory,
} from '../utils/documentTypes';

type ViewMode = 'grid' | 'table';
type SortBy = 'date' | 'size' | 'alpha';

export function DocumentsPage() {
  const { showError, showSuccess } = useToast();
  const [items, setItems] = useState<DocumentItem[]>([]);
  const [mockSamples, setMockSamples] = useState<DocumentItem[]>([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [bannerError, setBannerError] = useState('');
  const [view, setView] = useState<ViewMode>('grid');
  const [typeFilter, setTypeFilter] = useState<FileTypeCategory>('all');
  const [sortBy, setSortBy] = useState<SortBy>('date');

  const allItems = useMemo(
    () => [...items, ...mockSamples],
    [items, mockSamples],
  );

  const filtered = useMemo(() => {
    let list = filterByFileType(allItems, typeFilter).filter((i) =>
      i.name.toLowerCase().includes(query.toLowerCase()),
    );
    list = [...list].sort((a, b) => {
      if (sortBy === 'alpha') return a.name.localeCompare(b.name);
      if (sortBy === 'size') return b.blocks - a.blocks;
      return 0;
    });
    return list;
  }, [allItems, query, typeFilter, sortBy]);

  const load = () => {
    setLoading(true);
    setBannerError('');
    return api
      .documents()
      .then(setItems)
      .catch((e) => {
        const msg = e instanceof Error ? e.message : 'Could not load documents.';
        setBannerError(msg);
        showError(msg, load);
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    void load();
  }, []);

  async function upload(file?: File) {
    if (!file) return;
    try {
      await api.upload(file);
      showSuccess('Document uploaded', file.name);
      void load();
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Upload failed.';
      setBannerError(msg);
      showError(msg);
    }
  }

  async function remove(id: string) {
    if (id.startsWith('sample-')) {
      setMockSamples((prev) => {
        const next = prev.filter((i) => i.id !== id);
        sessionStorage.setItem('cb-mock-samples', JSON.stringify(next));
        window.dispatchEvent(new CustomEvent('cb:samples-updated'));
        return next;
      });
      return;
    }
    if (!confirm('Delete this document?')) return;
    try {
      await api.remove(id);
      void load();
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Delete failed.');
    }
  }

  function loadSamples() {
    setMockSamples(sampleDocuments);
    sessionStorage.setItem('cb-mock-samples', JSON.stringify(sampleDocuments));
    window.dispatchEvent(new CustomEvent('cb:samples-updated'));
    showSuccess('Sample docs loaded', '3 demo documents added for testing.');
  }

  const typeCounts = useMemo(() => countByFileType(allItems), [allItems]);
  const typeTabs = fileTypeTabs(typeCounts);

  return (
    <section className="page">
      <SurfaceCard className="page-card">
        <header className="page-card__header">
          <div>
            <h1>Documents</h1>
            <p>{allItems.length} prepared documents</p>
          </div>
        </header>

        {allItems.length === 0 && !loading && (
          <DragDropZone
            onFile={upload}
            label="Drag & drop PDF, DOCX, or CSV files here, or click to browse"
            sublabel={`Max ${maxFileSizeMb}MB per file`}
          />
        )}

        <input
          className="search-input"
          placeholder="Search documents..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <div className="docs-toolbar">
          <div className="filter-tabs">
            {typeTabs.map((tab) => (
              <button
                key={tab.key}
                type="button"
                className={typeFilter === tab.key ? 'active' : ''}
                onClick={() => setTypeFilter(tab.key)}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="docs-toolbar__right">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as SortBy)}
              aria-label="Sort documents"
            >
              <option value="date">Date Added</option>
              <option value="size">File Size</option>
              <option value="alpha">Alphabetical</option>
            </select>
            <div className="view-toggle">
              <button
                type="button"
                className={view === 'grid' ? 'active' : ''}
                onClick={() => setView('grid')}
                aria-label="Grid view"
              >
                <Grid3x3 size={16} />
              </button>
              <button
                type="button"
                className={view === 'table' ? 'active' : ''}
                onClick={() => setView('table')}
                aria-label="Table view"
              >
                <LayoutList size={16} />
              </button>
            </div>
          </div>
        </div>

        {bannerError && (
          <ErrorBanner
            message={bannerError}
            onRetry={load}
            onDismiss={() => setBannerError('')}
          />
        )}

        {loading ? (
          <SkeletonList count={4} />
        ) : (
          <div className={`document-list document-list--${view}`}>
            {filtered.map((i) => (
              <DocumentCard key={i.id} item={i} onDelete={remove} />
            ))}
          </div>
        )}

        {allItems.length === 0 && !loading && (
          <SurfaceCard className="sample-docs-card">
            <Sparkles size={20} />
            <div>
              <strong>Load Sample Docs</strong>
              <p>
                Don&apos;t have files ready? Load 3 sample tech docs to test
                search and chat.
              </p>
            </div>
            <button type="button" className="button" onClick={loadSamples}>
              Load samples
            </button>
          </SurfaceCard>
        )}

        <footer className="format-badges">
          {supportedFormats.map((fmt) => (
            <span key={fmt} className="format-badge">
              {fmt}
            </span>
          ))}
          <span className="format-badge format-badge--muted">
            Max {maxFileSizeMb}MB per file
          </span>
        </footer>
      </SurfaceCard>
    </section>
  );
}
