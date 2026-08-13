import React from 'react';
import ReactDOM from 'react-dom/client';
import { ClerkProvider, SignInButton, SignUpButton, Show, useAuth } from '@clerk/react';
import App from './App';
import { setTokenProvider } from './api/client';
import './styles/global.css';

function AuthenticatedApp(){const {getToken}=useAuth();React.useEffect(()=>{setTokenProvider(()=>getToken())},[getToken]);return <App/>}
function AuthScreen(){return <main className="auth-screen"><section className="surface-card"><h1>Corporate Brain</h1><p>Sign in to access company documents and your conversation history.</p><div><SignInButton mode="modal"><button className="button">Sign in</button></SignInButton><SignUpButton mode="modal"><button className="button secondary">Create account</button></SignUpButton></div></section></main>}
function Root(){return <><Show when="signed-out"><AuthScreen/></Show><Show when="signed-in"><AuthenticatedApp/></Show></>}

const key=import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode>{key?<ClerkProvider><Root/></ClerkProvider>:<main className="auth-screen"><section className="surface-card"><h1>Authentication required</h1><p>Set <code>VITE_CLERK_PUBLISHABLE_KEY</code> in <code>frontend/.env.local</code>, then restart Vite.</p></section></main>}</React.StrictMode>);
