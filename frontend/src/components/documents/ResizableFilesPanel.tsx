import {
  ChevronLeft,
  ChevronRight,
  FileSpreadsheet,
  FileText,
  Pin,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { DocumentItem } from '../../types';
import { uploadQuota } from '../../data/mockData';
import { DragDropZone } from '../ui/DragDropZone';
import { SkeletonList } from '../ui/Skeleton';

const MIN_WIDTH = 56;
const DEFAULT_WIDTH = 320;
type FileFilter = 'all' | 'pdf' | 'doc' | 'pinned';

export function ResizableFilesPanel({
  documents,
  active,
  onSelect,
  onUpload,
  containerRef,
  loading = false,
}: {
  documents: DocumentItem[];
  active: string;
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
  containerRef: React.RefObject<HTMLElement | null>;
  loading?: boolean;
}) {
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const [dragging, setDragging] = useState(false);
  const [filter, setFilter] = useState<FileFilter>('all');
  const [pinned, setPinned] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('cb-pinned-files') ?? '[]') as string[];
    } catch {
      return [];
    }
  });
  const inputRef = useRef<HTMLInputElement>(null);
  const collapsed = width <= MIN_WIDTH + 20;

  const visible = useMemo(() => {
    return documents.filter((doc) => {
      if (filter === 'pinned') return pinned.includes(doc.id);
      if (filter === 'pdf') return doc.type === 'pdf';
      if (filter === 'doc') return doc.type === 'docx' || doc.type === 'doc';
      return true;
    });
  }, [documents, filter, pinned]);

  const startDrag = useCallback(() => setDragging(true), []);

  useEffect(() => {
    if (!dragging) return;
    function onMove(e: MouseEvent) {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const maxWidth = rect.width * 0.5;
      const next = rect.right - e.clientX;
      setWidth(Math.max(MIN_WIDTH, Math.min(maxWidth, next)));
    }
    function onUp() {
      setDragging(false);
    }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [dragging, containerRef]);

  function selectDocument(id: string) {
    onSelect(id);
    if (collapsed) setWidth(DEFAULT_WIDTH);
  }

  function toggleCollapse() {
    setWidth(collapsed ? DEFAULT_WIDTH : MIN_WIDTH);
  }

  function togglePin(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    setPinned((prev) => {
      const next = prev.includes(id)
        ? prev.filter((p) => p !== id)
        : [...prev, id];
      localStorage.setItem('cb-pinned-files', JSON.stringify(next));
      return next;
    });
  }

  const filters: { key: FileFilter; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'pdf', label: 'PDFs' },
    { key: 'doc', label: 'Docs' },
    { key: 'pinned', label: 'Pinned' },
  ];

  return (
    <aside
      className={`files-panel glass-card${collapsed ? ' files-panel--collapsed' : ''}${dragging ? ' files-panel--dragging' : ''}`}
      style={{ width }}
    >
      <input
        ref={inputRef}
        hidden
        type="file"
        accept=".pdf,.docx,.doc,.xlsx,.csv,.zip,.txt"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onUpload(file);
          e.currentTarget.value = '';
        }}
      />
      <div
        className="files-panel__resize"
        onMouseDown={startDrag}
        aria-hidden="true"
      />
      <button
        className="files-panel__collapse"
        onClick={toggleCollapse}
        aria-label={collapsed ? 'Expand panel' : 'Collapse panel'}
        type="button"
      >
        {collapsed ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
      </button>

      {!collapsed && (
        <>
          <header className="files-panel__header">
            <h2>
              Uploaded Files <span>({documents.length})</span>
            </h2>
            <div className="quota-bar">
              <div className="quota-bar__track">
                <div
                  className="quota-bar__fill"
                  style={{
                    width: `${Math.min(100, (documents.length / uploadQuota.maxFiles) * 100)}%`,
                  }}
                />
              </div>
              <small>
                {documents.length} / {uploadQuota.maxFiles} files · 0 MB /{' '}
                {uploadQuota.maxMb} MB used
              </small>
            </div>
          </header>

          <div className="filter-tabs">
            {filters.map((f) => (
              <button
                key={f.key}
                type="button"
                className={filter === f.key ? 'active' : ''}
                onClick={() => setFilter(f.key)}
              >
                {f.label}
              </button>
            ))}
          </div>

          <div className="files-panel__list">
            {loading && <SkeletonList count={3} />}

            {!loading && documents.length === 0 && (
              <DragDropZone
                compact
                onFile={onUpload}
                label="Drag & drop files here or browse"
                sublabel="PDF, DOCX, CSV up to 25 MB"
              />
            )}

            {!loading &&
              documents.length > 0 &&
              visible.length === 0 && (
                <p className="files-panel__empty">
                  No files match this filter.
                </p>
              )}

            {!loading &&
              visible.map((doc) => (
                <button
                  key={doc.id}
                  className={`files-panel__item${active === doc.id ? ' active' : ''}`}
                  onClick={() => selectDocument(doc.id)}
                  type="button"
                >
                  <span className="files-panel__icon">
                    {doc.type === 'xlsx' ? (
                      <FileSpreadsheet size={16} />
                    ) : (
                      <FileText size={16} />
                    )}
                  </span>
                  <span className="files-panel__meta">
                    <strong>{doc.name}</strong>
                    <small>
                      {doc.type.toUpperCase()} · {doc.blocks} chunks ·{' '}
                      {doc.status.replaceAll('_', ' ')}
                    </small>
                  </span>
                  <span
                    className={`files-panel__pin${pinned.includes(doc.id) ? ' pinned' : ''}`}
                    onClick={(e) => togglePin(doc.id, e)}
                    role="button"
                    tabIndex={0}
                    aria-label="Pin file"
                  >
                    <Pin size={14} />
                  </span>
                </button>
              ))}
          </div>
        </>
      )}

      {collapsed && (
        <div className="files-panel__collapsed-icons">
          <button
            onClick={() => inputRef.current?.click()}
            title="Upload file"
            type="button"
          >
            <FileText size={18} />
          </button>
          <span className="files-panel__count">{documents.length}</span>
        </div>
      )}
    </aside>
  );
}
