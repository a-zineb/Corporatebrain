import type { ReactNode } from 'react';

export function Tooltip({
  content,
  children,
}: {
  content: string;
  children: ReactNode;
}) {
  return (
    <span className="tooltip-wrap">
      {children}
      <span className="tooltip-wrap__tip" role="tooltip">
        {content}
      </span>
    </span>
  );
}
