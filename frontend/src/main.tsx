import React from 'react';
import ReactDOM from 'react-dom/client';
import { ClerkProvider, useAuth } from '@clerk/react';
import App from './App';
import { setTokenProvider } from './api/client';
import './styles/global.css';

function ClerkAwareApp(){const {getToken,isLoaded}=useAuth();React.useEffect(()=>{setTokenProvider(async(forceRefresh=false)=>isLoaded?getToken({skipCache:forceRefresh}):null)},[getToken,isLoaded]);return <App/>}

const key = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY ?? '';
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {key ? (
      <ClerkProvider publishableKey={key}>
        <ClerkAwareApp />
      </ClerkProvider>
    ) : (
      <App />
    )}
  </React.StrictMode>,
);
