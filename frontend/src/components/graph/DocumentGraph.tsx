import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { DocumentItem } from '../../types';
import {
  buildDocumentGraph,
  filterGraphToFocus,
  simulateGraph,
  type GraphEdge,
  type GraphNode,
} from '../../utils/documentTypes';

export function DocumentGraph({ documents }: { documents: DocumentItem[] }) {
  const navigate = useNavigate();
  const svgRef = useRef<SVGSVGElement>(null);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const dragRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(
    null,
  );

  const fullGraph = useMemo(
    () => buildDocumentGraph(documents),
    [documents],
  );

  const layout = useMemo(() => {
    const { nodes, edges } = focusId
      ? filterGraphToFocus(fullGraph.nodes, fullGraph.edges, focusId)
      : fullGraph;
    if (nodes.length === 0) return { nodes: [], edges: [] };
    const simulated = simulateGraph(nodes, edges, 800, 520);
    return { nodes: simulated, edges };
  }, [fullGraph, focusId]);

  const hoverNode = layout.nodes.find((n) => n.id === hoverId);

  function onWheel(e: React.WheelEvent) {
    e.preventDefault();
    setZoom((z) => Math.min(2.5, Math.max(0.4, z - e.deltaY * 0.001)));
  }

  function onPointerDown(e: React.PointerEvent) {
    if (e.button !== 0) return;
    dragRef.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  }

  function onPointerMove(e: React.PointerEvent) {
    if (!dragRef.current) return;
    setPan({
      x: dragRef.current.panX + (e.clientX - dragRef.current.x),
      y: dragRef.current.panY + (e.clientY - dragRef.current.y),
    });
  }

  function onPointerUp() {
    dragRef.current = null;
  }

  function nodeRadius(n: GraphNode) {
    return 14 + n.connections * 4;
  }

  if (documents.length < 2) {
    return (
      <div className="graph-empty surface-card">
        <h2>Not enough documents</h2>
        <p>
          Upload at least 2 documents to see how they connect in Graph View.
        </p>
      </div>
    );
  }

  if (fullGraph.edges.length === 0) {
    return (
      <div className="graph-empty surface-card">
        <h2>No connections yet</h2>
        <p>
          Upload more documents to see connections — Graph View needs documents
          that share at least 5 keywords.
        </p>
      </div>
    );
  }

  return (
    <div className="graph-container">
      <div className="graph-toolbar">
        {focusId ? (
          <button type="button" className="chip" onClick={() => setFocusId(null)}>
            ← Back to Global View
          </button>
        ) : (
          <span className="graph-toolbar__hint">
            Global view · scroll to zoom · drag to pan · click node to focus
          </span>
        )}
        <button
          type="button"
          className="chip"
          onClick={() => {
            setPan({ x: 0, y: 0 });
            setZoom(1);
          }}
        >
          Reset view
        </button>
      </div>

      <svg
        ref={svgRef}
        className="graph-canvas"
        viewBox="0 0 800 520"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
          {layout.edges.map((e: GraphEdge) => {
            const a = layout.nodes.find((n) => n.id === e.source)!;
            const b = layout.nodes.find((n) => n.id === e.target)!;
            return (
              <line
                key={`${e.source}-${e.target}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                className="graph-edge"
                strokeWidth={Math.min(4, 1 + e.weight * 0.2)}
              />
            );
          })}
          {layout.nodes.map((n) => (
            <g
              key={n.id}
              transform={`translate(${n.x},${n.y})`}
              className={`graph-node${focusId === n.id ? ' graph-node--focus' : ''}`}
              onMouseEnter={() => setHoverId(n.id)}
              onMouseLeave={() => setHoverId(null)}
              onClick={() => setFocusId(n.id)}
              onDoubleClick={() => navigate('/')}
              style={{ cursor: 'pointer' }}
            >
              <circle r={nodeRadius(n)} className="graph-node__circle" />
              <text y={nodeRadius(n) + 14} textAnchor="middle" className="graph-node__label">
                {n.label.length > 22 ? `${n.label.slice(0, 20)}…` : n.label}
              </text>
            </g>
          ))}
        </g>
      </svg>

      {hoverNode && (
        <div className="graph-tooltip surface-card">
          <strong>{hoverNode.label}</strong>
          <span>{hoverNode.type.toUpperCase()}</span>
          <span>{hoverNode.connections} connection(s)</span>
          <small>Click to focus · Double-click to open chat</small>
        </div>
      )}
    </div>
  );
}
