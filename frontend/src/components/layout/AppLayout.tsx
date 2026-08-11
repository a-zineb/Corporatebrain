import {Outlet} from 'react-router-dom';import {Sidebar} from './Sidebar';
export function AppLayout({collapsed,onToggle}:{collapsed:boolean;onToggle:()=>void}){return <div className="app-shell"><Sidebar collapsed={collapsed} onToggle={onToggle}/><main><Outlet/></main></div>}
