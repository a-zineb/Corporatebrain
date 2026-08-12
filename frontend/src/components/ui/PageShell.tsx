import type { ReactNode } from 'react';

export function PageShell({
  children,
  className = '',
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`page-shell ${className}`.trim()}>
      <div className="page-shell__inner">{children}</div>
    </div>
  );
}

export function SurfaceCard({
  children,
  className = '',
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`surface-card ${className}`.trim()}>{children}</div>
  );
}
