const API = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
export async function request<T>(path:string, init?:RequestInit):Promise<T>{
  const response=await fetch(`${API}${path}`,init);
  if(!response.ok){let message="Corporate Brain couldn't process this request.";try{const data=await response.json() as {detail?:string};message=data.detail??message}catch{}throw new Error(message)}
  if(response.status===204)return undefined as T;
  return response.json() as Promise<T>;
}
