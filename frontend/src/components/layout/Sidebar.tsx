import { ChevronDown, ChevronRight, FileText, History, MessageSquare, Network, Settings } from 'lucide-react';
import { useEffect, useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { Show, SignInButton, SignUpButton, UserButton, useUser } from '@clerk/react';
import { api } from '../../api/corporateBrain';
import type { ConversationSummary, Theme } from '../../types';

export function Sidebar({theme,setTheme}:{theme:Theme;setTheme:(t:Theme)=>void}){
  const navigate=useNavigate();const {user}=useUser();const [recentOpen,setRecentOpen]=useState(true);const [recent,setRecent]=useState<ConversationSummary[]>([]);
  const isDark=theme==='dark'||(theme==='system'&&matchMedia('(prefers-color-scheme: dark)').matches);
  useEffect(()=>{const load=()=>api.conversations().then(items=>setRecent(items.slice(0,5))).catch(()=>setRecent([]));void load();window.addEventListener('cb:history-updated',load);return()=>window.removeEventListener('cb:history-updated',load)},[]);
  async function restore(id:string){const chat=await api.conversation(id);sessionStorage.setItem('cb-restore-chat',JSON.stringify(chat));navigate('/')}
  const link=(to:string,label:string,icon:React.ReactNode,end=false)=><NavLink to={to} end={end} className={({isActive})=>`sidebar__link${isActive?' active':''}`}>{icon}{label}</NavLink>;
  return <aside className="sidebar glass-card">
    <div className="sidebar__brand"><span className="sidebar__logo">Corporate Brain</span></div>
    <nav className="sidebar__nav">
      {link('/','AI Chat',<MessageSquare size={18}/>,true)}
      {recent.length>0&&<div className="sidebar__recent"><button type="button" className="sidebar__recent-toggle" onClick={()=>setRecentOpen(v=>!v)}>{recentOpen?<ChevronDown size={14}/>:<ChevronRight size={14}/>}Recent</button>{recentOpen&&recent.map(chat=><button key={chat.id} type="button" className="sidebar__recent-item" onClick={()=>void restore(chat.id)} title={chat.title}>{chat.title}</button>)}</div>}
      {link('/documents','Documents',<FileText size={18}/>)}
      {link('/graph','Graph View',<Network size={18}/>)}
      {link('/history','History',<History size={18}/>)}
      {link('/settings','Settings',<Settings size={18}/>)}
    </nav>
    <button type="button" className="sidebar__profile surface-card" onClick={()=>navigate('/profile')}>
      {user?.imageUrl?<img className="sidebar__avatar-img" src={user.imageUrl} alt=""/>:<span className="sidebar__avatar">{user?.firstName?.slice(0,1)??'?'}</span>}
      <div><strong>{user?.fullName??'Profile'}</strong><small>{user?.primaryEmailAddress?.emailAddress}</small></div>
    </button>
    <div className="sidebar__footer"><div className="theme-switch"><span>{isDark?'Dark':'Light'}</span><button className={`theme-switch__toggle${isDark?'':' on'}`} onClick={()=>setTheme(isDark?'light':'dark')} aria-label="Toggle theme" type="button"><span className="theme-switch__knob"/></button></div><Show when="signed-in"><UserButton/></Show><Show when="signed-out"><div><SignInButton mode="modal"><button type="button" className="button">Sign in</button></SignInButton><SignUpButton mode="modal"><button type="button" className="button secondary">Sign up</button></SignUpButton></div></Show></div>
  </aside>
}
