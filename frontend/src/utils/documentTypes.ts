import type { DocumentItem } from '../types';

export type FileTypeCategory = 'all' | 'pdf' | 'word' | 'csv' | 'other';

const STOP = new Set([
  'the','and','for','les','des','avec','pdf','doc','docx','file','fichier','document',
  'version','specification','project','projet','system','systeme','rapport','final','technique',
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
  radius: number;
  document: DocumentItem;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  reasons: string[];
}

export const GRAPH_SIMILARITY_MIN = 0.18;
export const GRAPH_MAX_EDGES_PER_NODE = 3;

export function buildDocumentGraph(documents: DocumentItem[]): {
  nodes: GraphNode[];
  edges: GraphEdge[];
} {
  const keywordMap = new Map<string, Set<string>>();
  documents.forEach((doc) => {
    keywordMap.set(doc.id, extractKeywords(doc,[doc.filiale??'',doc.application??''].filter(Boolean)));
  });

  const candidates: GraphEdge[] = [];
  for (let i = 0; i < documents.length; i++) {
    for (let j = i + 1; j < documents.length; j++) {
      const a = documents[i];
      const b = documents[j];
      const ka = keywordMap.get(a.id)!;
      const kb = keywordMap.get(b.id)!;
      const reasons=[...ka].filter(k=>kb.has(k));
      const union=new Set([...ka,...kb]).size;
      const score=union?reasons.length/union:0;
      if(score>=GRAPH_SIMILARITY_MIN&&reasons.length)candidates.push({source:a.id,target:b.id,weight:score,reasons:reasons.slice(0,6)});
    }
  }
  candidates.sort((a,b)=>b.weight-a.weight);
  const degree=new Map<string,number>();
  const edges=candidates.filter(edge=>{const a=degree.get(edge.source)??0,b=degree.get(edge.target)??0;if(a>=GRAPH_MAX_EDGES_PER_NODE||b>=GRAPH_MAX_EDGES_PER_NODE)return false;degree.set(edge.source,a+1);degree.set(edge.target,b+1);return true});

  const connectionCount = new Map<string, number>();
  documents.forEach((d) => connectionCount.set(d.id, 0));
  edges.forEach((e) => {
    connectionCount.set(e.source, (connectionCount.get(e.source) ?? 0) + 1);
    connectionCount.set(e.target, (connectionCount.get(e.target) ?? 0) + 1);
  });

  const nodes: GraphNode[] = documents.map((doc, i) => {
    const angle = (i / Math.max(documents.length, 1)) * Math.PI * 2;
    const r = 150 + Math.min(connectionCount.get(doc.id)!,4) * 12;
    const radius=Math.max(18,Math.min(46,22+Math.sqrt(Math.max(doc.blocks,1))*0.8));
    return {
      id: doc.id,
      label: doc.name,
      type: doc.type,
      connections: connectionCount.get(doc.id) ?? 0,
      x: 400 + Math.cos(angle) * r,
      y: 300 + Math.sin(angle) * r,
      vx: 0,
      vy: 0,
      radius,
      document:doc,
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
        const minimum=sim[i].radius+sim[j].radius+42;
        const force = dist<minimum?(minimum-dist)*0.32:9000/(dist*dist);
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
      const force = (dist-145) * 0.018;
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
      n.x = Math.max(n.radius+70, Math.min(width-n.radius-70,n.x));
      n.y = Math.max(n.radius+25, Math.min(height-n.radius-45,n.y));
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

export function fuzzyDocumentMatch(document:DocumentItem,query:string){
  const normalized=(value:string)=>value.toLocaleLowerCase().normalize('NFD').replace(/\p{Diacritic}/gu,'').replace(/[^a-z0-9]+/g,' ');
  const target=normalized(document.name),needle=normalized(query).trim();
  if(!needle||target.includes(needle))return true;
  return needle.split(' ').every(token=>target.split(' ').some(word=>{
    if(word.startsWith(token)||token.startsWith(word))return true;
    if(Math.abs(word.length-token.length)>2)return false;
    let differences=0,i=0,j=0;while(i<word.length&&j<token.length){if(word[i]===token[j]){i++;j++}else{differences++;if(word.length>token.length)i++;else if(token.length>word.length)j++;else{i++;j++}}}return differences+(word.length-i)+(token.length-j)<=2;
  }));
}
