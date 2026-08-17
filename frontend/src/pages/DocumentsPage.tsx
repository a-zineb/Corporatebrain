import { useEffect, useMemo, useState } from 'react';
import {
  Grid3x3,
  LayoutList,
} from 'lucide-react';
import { api } from '../api/corporateBrain';
import type { DocumentItem, IngestionJob } from '../types';
import { DocumentCard } from '../components/documents/DocumentCard';
import { DragDropZone } from '../components/ui/DragDropZone';
import { ErrorBanner } from '../components/ui/ErrorBanner';
import { SurfaceCard } from '../components/ui/PageShell';
import { SkeletonList } from '../components/ui/Skeleton';
import { useToast } from '../hooks/useToast';
import {
  maxFileSizeMb,
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
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [bannerError, setBannerError] = useState('');
  const [view, setView] = useState<ViewMode>('grid');
  const [typeFilter, setTypeFilter] = useState<FileTypeCategory>('all');
  const [sortBy, setSortBy] = useState<SortBy>('date');
  const [jobs,setJobs]=useState<IngestionJob[]>([]);

  const allItems = items;

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

  useEffect(()=>{const poll=()=>api.ingestionJobs().then(next=>{setJobs(next);if(next.some(job=>job.status==='complete'))void load()}).catch(()=>undefined);void poll();const timer=setInterval(poll,1000);return()=>clearInterval(timer)},[]);

  async function upload(file?: File) {
    if (!file) return;
    try {
      await api.uploadAsync(file);
      showSuccess('Upload received', `${file.name} is processing in the background.`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Upload failed.';
      setBannerError(msg);
      showError(msg);
    }
  }

  async function reindex(id:string){try{await api.reindex(id);showSuccess('Document re-indexed','Old chunks were replaced for this document only.');void load()}catch(e){showError(e instanceof Error?e.message:'Re-index failed.')}}

  async function retry(id:string){try{await api.retryIngestion(id)}catch(e){showError(e instanceof Error?e.message:'Retry failed.')}}

  async function remove(id: string) {
    if (!confirm('Delete this document?')) return;
    try {
      await api.remove(id);
      void load();
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Delete failed.');
    }
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

        {jobs.length>0&&<section className="ingestion-progress" aria-label="Ingestion progress"><header><strong>Processing files</strong><span>{jobs.filter(job=>job.status==='complete').length}/{jobs.length} complete</span></header>{jobs.map(job=><div key={job.id} className={`ingestion-job ingestion-job--${job.status}`}><div><strong>{job.name}</strong><small>{job.stage}{job.units_total?` · ${job.units_completed}/${job.units_total}`:''}</small></div><progress max={job.total_stages} value={job.completed_stages}/>{job.status==='failed'&&<><span className="error">{job.error}</span><button onClick={()=>void retry(job.id)}>Retry</button></>}</div>)}</section>}

        {loading ? (
          <SkeletonList count={4} />
        ) : (
          <div className={`document-list document-list--${view}`}>
            {filtered.map((i) => (
              <DocumentCard key={i.id} item={i} onDelete={remove} onReindex={reindex} />
            ))}
          </div>
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
