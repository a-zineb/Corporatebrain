import {API_BASE,request} from './client'; import type {ChatResponse,DocumentItem,SearchResult,Source} from '../types';
export const api={
  health:()=>request<{status:string;service:string}>('/api/health'),
  documents:()=>request<DocumentItem[]>('/api/documents'),
  chat:(message:string,document_hash:string|undefined,mode:'direct'|'ai',conversation_id?:string)=>request<ChatResponse>('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,document_hash,mode,conversation_id})}),
  search:(query:string)=>request<{results:SearchResult[]}>('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,limit:100})}),
  upload:(file:File)=>{const body=new FormData();body.append('file',file);return request<DocumentItem>('/api/documents/upload',{method:'POST',body})},
  remove:(id:string)=>request<void>(`/api/documents/${id}`,{method:'DELETE'}),
  source:(hash:string,block:string)=>request<Source>(`/api/sources/${hash}/${block}`),
  documentSource:(hash:string)=>request<Source>(`/api/documents/${hash}/source`),
  originalUrl:(hash:string,download=false)=>`${API_BASE}/api/documents/${hash}/content${download?'?download=true':''}`
};
