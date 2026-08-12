import { Upload } from 'lucide-react';
import { useRef, useState, type DragEvent } from 'react';

export function DragDropZone({
  onFile,
  accept = '.pdf,.docx,.doc,.xlsx,.csv,.zip,.txt',
  label = 'Drag & drop files here or browse',
  sublabel,
  compact = false,
}: {
  onFile: (file: File) => void;
  accept?: string;
  label?: string;
  sublabel?: string;
  compact?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (file) onFile(file);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  }

  return (
    <div
      className={`drop-zone${dragging ? ' drop-zone--active' : ''}${compact ? ' drop-zone--compact' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      role="button"
      tabIndex={0}
    >
      <input
        ref={inputRef}
        hidden
        type="file"
        accept={accept}
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = '';
        }}
      />
      <span className="drop-zone__icon">
        <Upload size={compact ? 22 : 28} />
      </span>
      <p className="drop-zone__label">{label}</p>
      {sublabel && <p className="drop-zone__sublabel">{sublabel}</p>}
    </div>
  );
}
