export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
let tokenProvider: (() => Promise<string | null>) | undefined;
export function setTokenProvider(provider: () => Promise<string | null>) { tokenProvider = provider; }
export async function request<T>(path:string, init?:RequestInit):Promise<T>{
  const headers=new Headers(init?.headers);const token=await tokenProvider?.();if(token)headers.set('Authorization',`Bearer ${token}`);
  const response=await fetch(`${API_BASE}${path}`,{...init,headers});
  if(!response.ok){let message="Corporate Brain couldn't process this request.";try{const data=await response.json() as {detail?:string};message=data.detail??message}catch{}throw new Error(message)}
  if(response.status===204)return undefined as T;
  return response.json() as Promise<T>;
}
