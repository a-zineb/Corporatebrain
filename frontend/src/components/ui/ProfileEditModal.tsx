import { useRef, useState } from 'react';
import { X } from 'lucide-react';
import type { UserProfile } from '../../hooks/useUserProfile';

export function ProfileEditModal({
  profile,
  onSave,
  onClose,
}: {
  profile: UserProfile;
  onSave: (updates: Partial<UserProfile>) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState(profile.name);
  const [avatarUrl, setAvatarUrl] = useState(profile.avatarUrl);
  const picker = useRef<HTMLInputElement>(null);

  function handleAvatar(file?: File) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setAvatarUrl(String(reader.result));
    reader.readAsDataURL(file);
  }

  function save() {
    onSave({ name: name.trim() || profile.name, avatarUrl });
    onClose();
  }

  const initials = name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? '')
    .join('');

  return (
    <div className="modal-overlay" onClick={onClose} role="presentation">
      <div
        className="modal surface-card"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Edit profile"
      >
        <header className="modal__header">
          <h2>Edit Profile</h2>
          <button type="button" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </header>

        <div className="modal__body">
          <button
            type="button"
            className="profile-avatar-edit"
            onClick={() => picker.current?.click()}
          >
            {avatarUrl ? (
              <img src={avatarUrl} alt="" />
            ) : (
              <span>{initials || profile.initials}</span>
            )}
          </button>
          <input
            ref={picker}
            hidden
            type="file"
            accept="image/*"
            onChange={(e) => handleAvatar(e.target.files?.[0])}
          />
          <p className="profile-avatar-hint">Click to change avatar</p>

          <label className="modal__field">
            Display name
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your name"
            />
          </label>

          <p className="modal__meta">
            Plan: <strong>{profile.plan}</strong>
          </p>
        </div>

        <footer className="modal__footer">
          <button type="button" className="button button--ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="button" onClick={save}>
            Save
          </button>
        </footer>
      </div>
    </div>
  );
}
