export interface DocumentItem { id:string; name:string; type:string; status:string; blocks:number; warnings:string[]; filiale?:string|null; application?:string|null }
export interface IngestionJob { id:string;name:string;stage:'uploading'|'extracting'|'normalizing'|'chunking'|'embedding'|'indexing'|'ready'|'warning'|'failed';completed_stages:number;total_stages:number;status:'running'|'complete'|'failed';error?:string|null;document_id?:string|null;units_total?:number;units_completed?:number }
export interface Source { document:string; file_hash:string; file_type:string; block_id:string; location:string; page:number|null; sheet:string|null; row:number|null; row_end:number|null; cell_range:string|null; section:string|null; text:string; metadata:Record<string, unknown> }
export interface ChatResponse { answer:string; status:string; result_type:string; language:string; method:string; conversation_id:string; sources:Source[]; suggestions:string[]; latency_ms:number }
export interface HealthStatus { status:string; service:string; ai_provider_configured:boolean }
export interface SearchResult { document_hash:string; document_name:string; file_type:string; title:string; relation:string; entity:string; value:string; preview:string; score:number; source:Source }
export type Theme = 'system'|'dark'|'light';
export interface ConversationSummary { id:string; title:string; document_hash?:string; document_name?:string; created_at:string; updated_at:string }
export interface SavedConversation extends ConversationSummary { messages:Array<{role:string;text:string;response?:ChatResponse}> }
