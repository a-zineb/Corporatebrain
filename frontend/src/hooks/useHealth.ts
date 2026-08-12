import { useCallback, useEffect, useState } from 'react';
import { api } from '../api/corporateBrain';

export type ConnectionState = 'connecting' | 'ready' | 'error';

export function useHealth(pollMs = 30000) {
  const [state, setState] = useState<ConnectionState>('connecting');

  const check = useCallback(async () => {
    setState('connecting');
    try {
      const res = await api.health();
      setState(res.status === 'ok' ? 'ready' : 'error');
    } catch {
      setState('error');
    }
  }, []);

  useEffect(() => {
    void check();
    const id = setInterval(() => void check(), pollMs);
    return () => clearInterval(id);
  }, [check, pollMs]);

  return { state, retry: check };
}
