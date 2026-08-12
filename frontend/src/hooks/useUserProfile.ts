import { useCallback, useEffect, useState } from 'react';
import { userProfile as defaultProfile } from '../data/mockData';

export interface UserProfile {
  name: string;
  email: string;
  plan: string;
  initials: string;
  avatarUrl?: string;
}

const STORAGE_KEY = 'cb-user-profile';

function computeInitials(name: string) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? '')
    .join('');
}

function loadProfile(): UserProfile {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as UserProfile;
  } catch {
    /* use default */
  }
  return { ...defaultProfile };
}

export function useUserProfile() {
  const [profile, setProfile] = useState<UserProfile>(loadProfile);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
    window.dispatchEvent(new CustomEvent('cb:profile-updated'));
  }, [profile]);

  const saveProfile = useCallback((updates: Partial<UserProfile>) => {
    setProfile((prev) => {
      const name = updates.name ?? prev.name;
      const initials =
        updates.initials ??
        (updates.name ? computeInitials(name) : prev.initials);
      return { ...prev, ...updates, name, initials };
    });
  }, []);

  return { profile, saveProfile, setProfile };
}

export function useProfileSnapshot() {
  const [profile, setProfile] = useState<UserProfile>(loadProfile);

  useEffect(() => {
    const refresh = () => setProfile(loadProfile());
    window.addEventListener('cb:profile-updated', refresh);
    return () => window.removeEventListener('cb:profile-updated', refresh);
  }, []);

  return profile;
}
