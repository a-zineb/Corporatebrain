import { FormEvent, useEffect, useMemo, useState } from 'react';
import { Lightbulb, Network, Search } from 'lucide-react';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../api/corporateBrain';
import type { DocumentItem, SearchResult, Source } from '../types';
import { SourcePanel } from '../components/chat/SourcePanel';
import { ErrorBanner } from '../components/ui/ErrorBanner';
import { SurfaceCard } from '../components/ui/PageShell';
import { SkeletonCards } from '../components/ui/Skeleton';
import { useToast } from '../hooks/useToast';
import { searchTips } from '../data/mockData';
import {
  categorizeFileType,
  countByFileType,
  fileTypeChips,
  type FileTypeCategory,
} from '../utils/documentTypes';

export function SearchPage() {
  const { showError } = useToast();
  const [params, setParams] = useSearchParams();
  const initial = params.get('q') ?? '';
  const [query, setQuery] = useState(initial);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [bannerError, setBannerError] = useState('');
  const [source, setSource] = useState<Source | null>(null);
  const [page, setPage] = useState(0);
  const [typeFilter, setTypeFilter] = useState<FileTypeCategory>('all');
  const [strategy, setStrategy] = useState('hybrid');
  const [dateRange, setDateRange] = useState('any');
  const [catalogDocs, setCatalogDocs] = useState<DocumentItem[]>([]);
  const pageSize = 12;

  useEffect(() => {
    void api
      .documents()
      .then(setCatalogDocs)
      .catch(() => setCatalogDocs([]));
    try {
      const samples = sessionStorage.getItem('cb-mock-samples');
      if (samples) {
        setCatalogDocs((prev) => [
          ...prev,
          ...(JSON.parse(samples) as DocumentItem[]),
        ]);
      }
    } catch {
      /* ignore */
    }
    const onSamples = () => {
      try {
        const samples = sessionStorage.getItem('cb-mock-samples');
        if (samples) {
          void api.documents().then((docs) => {
            setCatalogDocs([...docs, ...(JSON.parse(samples) as DocumentItem[])]);
          });
        }
      } catch {
        /* ignore */
      }
    };
    window.addEventListener('cb:samples-updated', onSamples);
    return () => window.removeEventListener('cb:samples-updated', onSamples);
  }, []);

  const typeCounts = useMemo(() => countByFileType(catalogDocs), [catalogDocs]);
  const chips = useMemo(() => fileTypeChips(typeCounts), [typeCounts]);

  async function run(value = query) {
    if (!value.trim()) return;
    setBusy(true);
    setBannerError('');
    setParams({ q: value.trim() });
    try {
      setResults((await api.search(value.trim())).results);
      setPage(0);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Search failed.';
      setBannerError(msg);
      showError(msg, () => void run(value));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (initial) void run(initial);
  }, []);

  const filteredResults = useMemo(() => {
    if (typeFilter === 'all') return results;
    return results.filter(
      (r) => categorizeFileType(r.file_type) === typeFilter,
    );
  }, [results, typeFilter]);

  const common = useMemo(() => {
    const counts = new Map<string, number>();
    filteredResults.forEach((r) => {
      if (r.value) counts.set(r.value, (counts.get(r.value) ?? 0) + 1);
    });
    return [...counts].sort((a, b) => b[1] - a[1]).slice(0, 4);
  }, [filteredResults]);

  if (source) {
    return <SourcePanel source={source} onClose={() => setSource(null)} />;
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    void run();
  }

  function selectTypeFilter(category: FileTypeCategory) {
    setTypeFilter(category);
    setPage(0);
    if (!query.trim()) {
      const label =
        category === 'pdf'
          ? 'pdf'
          : category === 'word'
            ? 'docx'
            : category === 'csv'
              ? 'csv'
              : '';
      if (label) setQuery(label);
    }
  }

  return (
    <section className="page search-page">
      <SurfaceCard className="page-card">
        <header>
          <span className="eyebrow">Global Knowledge Search</span>
          <h1>Find anything, anywhere</h1>
          <p>
            Search every prepared document without changing your active chat
            document.
          </p>
          <Link to="/graph" className="search-graph-link">
            <Network size={16} /> View as Graph
          </Link>
        </header>

        <form className="search-box" onSubmit={submit}>
          <Search />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search documents, fields and sections..."
          />
          <button type="submit">{busy ? 'Searching…' : 'Search'}</button>
        </form>

        <div className="search-filters">
          <label>
            Document Type
            <select
              value={typeFilter}
              onChange={(e) => {
                setTypeFilter(e.target.value as FileTypeCategory);
                setPage(0);
              }}
            >
              <option value="all">All types ({typeCounts.all})</option>
              <option value="pdf">PDF ({typeCounts.pdf})</option>
              <option value="word">Word ({typeCounts.word})</option>
              <option value="csv">CSV ({typeCounts.csv})</option>
              <option value="other">Other ({typeCounts.other})</option>
            </select>
          </label>
          <label>
            Match Strategy
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
            >
              <option value="hybrid">Hybrid (BM25 + Vector)</option>
              <option value="semantic">Semantic only</option>
              <option value="keyword">Keyword only</option>
            </select>
          </label>
          <label>
            Date Range
            <select
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
            >
              <option value="any">Any time</option>
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="year">This year</option>
            </select>
          </label>
        </div>

        <div className="chip-row">
          {chips.map((chip) => (
            <button
              key={chip.key}
              type="button"
              className={`chip${typeFilter === chip.key ? ' active' : ''}`}
              onClick={() => {
                selectTypeFilter(chip.key);
                if (query.trim()) void run(query);
              }}
            >
              {chip.label} ({chip.count})
            </button>
          ))}
        </div>

        {bannerError && (
          <ErrorBanner
            message={bannerError}
            onRetry={() => void run()}
            onDismiss={() => setBannerError('')}
          />
        )}

        {filteredResults.length > 0 && (
          <div className="search-summary">
            <strong>{filteredResults.length} relevant matches</strong>
            {typeFilter !== 'all' && results.length !== filteredResults.length && (
              <span>
                Filtered from {results.length} total ·{' '}
                <button
                  type="button"
                  className="link-button"
                  onClick={() => setTypeFilter('all')}
                >
                  Clear type filter
                </button>
              </span>
            )}
            {common.length > 0 && (
              <span>
                Most common:{' '}
                {common.map(([value, count]) => `${value} (${count})`).join(' · ')}
              </span>
            )}
          </div>
        )}

        {busy && <SkeletonCards count={3} />}

        <div className="result-grid">
          {filteredResults
            .slice(page * pageSize, (page + 1) * pageSize)
            .map((r, i) => (
              <article
                className="result-card surface-card"
                key={`${r.document_hash}-${r.source.block_id}-${i}`}
              >
                <div>
                  <span className="badge">{r.file_type.toUpperCase()}</span>
                  <small>
                    {r.document_name} · {r.source.location}
                  </small>
                </div>
                <h3>{r.title}</h3>
                <strong>
                  {r.relation}: {r.value}
                </strong>
                <p>{r.preview}</p>
                <button type="button" onClick={() => setSource(r.source)}>
                  Open source
                </button>
              </article>
            ))}
        </div>

        {filteredResults.length > pageSize && (
          <div className="pagination">
            <button
              type="button"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </button>
            <span>
              {page * pageSize + 1}–
              {Math.min((page + 1) * pageSize, filteredResults.length)} of{' '}
              {filteredResults.length}
            </span>
            <button
              type="button"
              disabled={(page + 1) * pageSize >= filteredResults.length}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        )}

        {filteredResults.length === 0 && !busy && (
          <SurfaceCard className="tips-card">
            <h2>
              <Lightbulb size={18} /> Search tips
            </h2>
            <ul>
              {searchTips.map((tip) => (
                <li key={tip}>{tip}</li>
              ))}
            </ul>
          </SurfaceCard>
        )}
      </SurfaceCard>
    </section>
  );
}
