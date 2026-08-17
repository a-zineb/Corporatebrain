import { Clock, FileText, MessageSquare, Pencil, Trash2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/corporateBrain';
import type { ConversationSummary } from '../types';
import { SurfaceCard } from '../components/ui/PageShell';

export function HistoryPage(){const navigate=useNavigate();const [chats,setChats]=useState<ConversationSummary[]>([]);const [query,setQuery]=useState('');const [error,setError]=useState('');
  useEffect(()=>{api.conversations().then(setChats).catch(e=>setError(e instanceof Error?e.message:'Could not load history.'))},[]);
  async function restore(id:string){const chat=await api.conversation(id);sessionStorage.setItem('cb-restore-chat',JSON.stringify(chat));navigate('/')}
  async function remove(id:string){if(!confirm('Delete this conversation?'))return;await api.deleteConversation(id);setChats(items=>items.filter(item=>item.id!==id))}
  async function rename(chat:ConversationSummary){const title=prompt('Conversation title',chat.title)?.trim();if(!title||title===chat.title)return;await api.renameConversation(chat.id,title);setChats(items=>items.map(item=>item.id===chat.id?{...item,title}:item))}
  const visible=chats.filter(chat=>`${chat.title} ${chat.document_name??''}`.toLowerCase().includes(query.toLowerCase()));
  return <section className="page"><SurfaceCard className="page-card"><header><h1>History</h1><p>Your authenticated, private conversations.</p></header>
    <div className="history-toolbar"><input className="search-input" placeholder="Search history" value={query} onChange={e=>setQuery(e.target.value)}/></div>
    {error&&<p className="error">{error}</p>}{visible.length?<div className="history-list">{visible.map(chat=><div key={chat.id} className="history-card surface-card"><button onClick={()=>void restore(chat.id)}><Clock/><div><strong>{chat.title}</strong><small>{new Date(chat.updated_at).toLocaleString()}</small>{chat.document_name&&<span><FileText size={13}/>{chat.document_name}</span>}</div></button><button aria-label={`Rename ${chat.title}`} onClick={()=>void rename(chat)}><Pencil size={16}/></button><button aria-label={`Delete ${chat.title}`} onClick={()=>void remove(chat.id)}><Trash2 size={16}/></button></div>)}</div>:<div className="empty-state surface-card"><MessageSquare size={32}/><h2>No conversations yet</h2><p>Start a document chat to build your history.</p></div>}
  </SurfaceCard></section>}
