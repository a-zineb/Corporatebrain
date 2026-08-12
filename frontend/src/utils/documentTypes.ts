import type { DocumentItem } from '../types';

export type FileTypeCategory = 'all' | 'pdf' | 'word' | 'csv' | 'other';

const STOP = new Set([
  'the', 'and', 'for', 'pdf', 'doc', 'docx', 'file', 'document',
]);

export function categorizeFileType(type: string): Exclude<FileTypeCategory, 'all'> {
  const t = type.toLowerCase();
  if (t === 'pdf') return 'pdf';
  if (t === 'docx' || t === 'doc') return 'word';
  if (t === 'csv') return 'csv';
  return 'other';
}

export function countByFileType(documents: DocumentItem[]) {
  const counts = { all: documents.length, pdf: 0, word: 0, csv: 0, other: 0 };
  documents.forEach((doc) => {
    counts[categorizeFileType(doc.type)] += 1;
  });
  return counts;
}

export function filterByFileType(
  documents: DocumentItem[],
  category: FileTypeCategory,
) {
  if (category === 'all') return documents;
  return documents.filter((doc) => categorizeFileType(doc.type) === category);
}

export function fileTypeTabs(counts: ReturnType<typeof countByFileType>) {
  return [
    { key: 'all' as const, label: `All (${counts.all})` },
    { key: 'pdf' as const, label: `PDF (${counts.pdf})` },
    { key: 'word' as const, label: `Word (${counts.word})` },
    { key: 'csv' as const, label: `CSV (${counts.csv})` },
    { key: 'other' as const, label: `Other (${counts.other})` },
  ];
}

export function fileTypeChips(counts: ReturnType<typeof countByFileType>) {
  return [
    { key: 'pdf' as const, label: '📄 PDF', count: counts.pdf },
    { key: 'word' as const, label: '📝 Word', count: counts.word },
    { key: 'csv' as const, label: '📊 CSV', count: counts.csv },
    { key: 'other' as const, label: '📁 Other', count: counts.other },
  ].filter((c) => c.count > 0 || counts.all === 0);
}

/** Tokenize a document name into keyword candidates. */
export function extractKeywords(doc: DocumentItem, extra: string[] = []): Set<string> {
  const base = doc.name
    .replace(/\.[a-z0-9]+$/i, '')
    .split(/[^a-zA-Z0-9]+/)
    .map((w) => w.toLowerCase())
    .filter((w) => w.length >= 3 && !STOP.has(w));

  const typeTokens = doc.type === 'docx' ? ['word', 'document', 'policy', 'guide'] : [];
  const blockTokens =
    doc.blocks >= 30
      ? ['analysis', 'report', 'data', 'content', 'section']
      : ['summary', 'brief', 'notes'];

  return new Set([...base, ...extra, ...typeTokens, ...blockTokens]);
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  connections: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

const KEYWORD_MIN = 5;

export function buildDocumentGraph(documents: DocumentItem[]): {
  nodes: GraphNode[];
  edges: GraphEdge[];
} {
  const extras: Record<string, string[]> = {
    'sample-1': ['financial', 'results', 'quarter', 'revenue', 'report', 'annual'],
    'sample-2': ['security', 'compliance', 'policy', 'access', 'control', 'audit'],
    'sample-3': ['authentication', 'api', 'endpoint', 'security', 'token', 'oauth'],
  };

  const keywordMap = new Map<string, Set<string>>();
  documents.forEach((doc) => {
    keywordMap.set(doc.id, extractKeywords(doc, extras[doc.id] ?? []));
  });

  const edges: GraphEdge[] = [];
  for (let i = 0; i < documents.length; i++) {
    for (let j = i + 1; j < documents.length; j++) {
      const a = documents[i];
      const b = documents[j];
      const ka = keywordMap.get(a.id)!;
      const kb = keywordMap.get(b.id)!;
      let shared = 0;
      ka.forEach((k) => {
        if (kb.has(k)) shared += 1;
      });
      if (shared >= KEYWORD_MIN) {
        edges.push({ source: a.id, target: b.id, weight: shared });
      }
    }
  }

  const connectionCount = new Map<string, number>();
  documents.forEach((d) => connectionCount.set(d.id, 0));
  edges.forEach((e) => {
    connectionCount.set(e.source, (connectionCount.get(e.source) ?? 0) + 1);
    connectionCount.set(e.target, (connectionCount.get(e.target) ?? 0) + 1);
  });

  const nodes: GraphNode[] = documents.map((doc, i) => {
    const angle = (i / Math.max(documents.length, 1)) * Math.PI * 2;
    const r = 120 + connectionCount.get(doc.id)! * 18;
    return {
      id: doc.id,
      label: doc.name,
      type: doc.type,
      connections: connectionCount.get(doc.id) ?? 0,
      x: 400 + Math.cos(angle) * r,
      y: 300 + Math.sin(angle) * r,
      vx: 0,
      vy: 0,
    };
  });

  return { nodes, edges };
}

export function simulateGraph(
  nodes: GraphNode[],
  edges: GraphEdge[],
  width: number,
  height: number,
  steps = 120,
) {
  const sim = nodes.map((n) => ({ ...n }));
  const nodeMap = new Map(sim.map((n) => [n.id, n]));

  for (let s = 0; s < steps; s++) {
    for (let i = 0; i < sim.length; i++) {
      for (let j = i + 1; j < sim.length; j++) {
        const dx = sim[j].x - sim[i].x;
        const dy = sim[j].y - sim[i].y;
        const dist = Math.max(Math.hypot(dx, dy), 1);
        const force = 8000 / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        sim[i].vx -= fx;
        sim[i].vy -= fy;
        sim[j].vx += fx;
        sim[j].vy += fy;
      }
    }
    edges.forEach((e) => {
      const a = nodeMap.get(e.source)!;
      const b = nodeMap.get(e.target)!;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.max(Math.hypot(dx, dy), 1);
      const force = dist * 0.02;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    });
    sim.forEach((n) => {
      n.vx += (width / 2 - n.x) * 0.001;
      n.vy += (height / 2 - n.y) * 0.001;
      n.vx *= 0.85;
      n.vy *= 0.85;
      n.x += n.vx;
      n.y += n.vy;
      n.x = Math.max(40, Math.min(width - 40, n.x));
      n.y = Math.max(40, Math.min(height - 40, n.y));
    });
  }
  return sim;
}

export function filterGraphToFocus(
  nodes: GraphNode[],
  edges: GraphEdge[],
  focusId: string,
) {
  const linked = new Set<string>([focusId]);
  edges.forEach((e) => {
    if (e.source === focusId) linked.add(e.target);
    if (e.target === focusId) linked.add(e.source);
  });
  return {
    nodes: nodes.filter((n) => linked.has(n.id)),
    edges: edges.filter(
      (e) => linked.has(e.source) && linked.has(e.target),
    ),
  };
}
