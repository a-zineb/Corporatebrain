import {FileText,History,MessageSquarePlus,PanelLeftClose,PanelLeftOpen,Search,Settings} from 'lucide-react';
import {NavLink} from 'react-router-dom';
const items=[['/','New Chat',MessageSquarePlus],['/documents','Documents',FileText],['/search','Search',Search],['/history','History',History],['/settings','Settings',Settings]] as const;
export function Sidebar({collapsed,onToggle}:{collapsed:boolean;onToggle:()=>void}){return <aside className={`sidebar ${collapsed?'collapsed':''}`}>
  <div className="brand"><span className="orb small"/><span className="label">Corporate Brain</span></div>
  <nav>{items.map(([to,label,Icon])=><NavLink key={to} to={to} title={collapsed?label:undefined}><Icon size={19}/><span className="label">{label}</span></NavLink>)}</nav>
  <button className="collapse" onClick={onToggle} aria-label="Toggle sidebar">{collapsed?<PanelLeftOpen/>:<PanelLeftClose/>}<span className="label">Collapse</span></button>
</aside>}
