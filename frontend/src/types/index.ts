export interface DocumentItem { id:string; name:string; type:string; status:string; blocks:number; warnings:string[] }
export interface Source { document:string; file_hash:string; file_type:string; block_id:string; location:string; page:number|null; sheet:string|null; row:number|null; section:string|null; text:string; metadata:Record<string, unknown> }
export interface ChatResponse { answer:string; status:string; result_type:string; language:string; method:string; conversation_id:string; sources:Source[]; suggestions:string[]; latency_ms:number }
export interface SearchResult { document_hash:string; document_name:string; file_type:string; title:string; relation:string; entity:string; value:string; preview:string; score:number; source:Source }
export type Theme = 'system'|'dark'|'light';
