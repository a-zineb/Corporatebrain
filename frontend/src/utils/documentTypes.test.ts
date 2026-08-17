import {describe,expect,it} from 'vitest';
import type {DocumentItem} from '../types';
import {buildDocumentGraph,filterGraphToFocus,fuzzyDocumentMatch,GRAPH_MAX_EDGES_PER_NODE,simulateGraph} from './documentTypes';

const doc=(id:string,name:string,type='pdf',blocks=100):DocumentItem=>({id,name,type,blocks,status:'READY',warnings:[],filiale:'OCM',application:'MZ'});
const documents=[doc('a','OCM P2P collection.pdf','pdf',1),doc('b','OCM P2P workflow.docx','docx',100),doc('c','OCM P2P CDR.xlsx','xlsx',10000),doc('d','Unrelated local logistics.csv','csv',20)];

describe('document graph',()=>{
  it('has no self loops or duplicate edges and caps degree',()=>{const graph=buildDocumentGraph(documents);const keys=graph.edges.map(e=>[e.source,e.target].sort().join(':'));expect(new Set(keys).size).toBe(keys.length);expect(graph.edges.every(e=>e.source!==e.target)).toBe(true);for(const node of graph.nodes)expect(graph.edges.filter(e=>e.source===node.id||e.target===node.id).length).toBeLessThanOrEqual(GRAPH_MAX_EDGES_PER_NODE)});
  it('bounds node radii',()=>expect(buildDocumentGraph(documents).nodes.every(n=>n.radius>=18&&n.radius<=46)).toBe(true));
  it('filters focus to direct neighbors',()=>{const graph=buildDocumentGraph(documents);const focused=filterGraphToFocus(graph.nodes,graph.edges,'a');expect(focused.nodes.every(n=>n.id==='a'||graph.edges.some(e=>(e.source==='a'&&e.target===n.id)||(e.target==='a'&&e.source===n.id)))).toBe(true)});
  it('supports accent-insensitive approximate search',()=>{expect(fuzzyDocumentMatch(doc('x','Spécification (P2P).pdf'),'specifcation p2p')).toBe(true)});
  it('separates node circles after simulation',()=>{const graph=buildDocumentGraph(documents.slice(0,3));const nodes=simulateGraph(graph.nodes,graph.edges,1000,620,220);for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++)expect(Math.hypot(nodes[i].x-nodes[j].x,nodes[i].y-nodes[j].y)).toBeGreaterThanOrEqual(nodes[i].radius+nodes[j].radius-2)});
});
