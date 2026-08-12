export function Skeleton({
  className = '',
  style,
}: {
  className?: string;
  style?: React.CSSProperties;
}) {
  return <div className={`skeleton ${className}`} style={style} aria-hidden="true" />;
}

export function SkeletonList({ count = 4 }: { count?: number }) {
  return (
    <div className="skeleton-list">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="skeleton-row glass-card">
          <Skeleton className="skeleton-row__avatar" />
          <div className="skeleton-row__lines">
            <Skeleton className="skeleton-row__line skeleton-row__line--wide" />
            <Skeleton className="skeleton-row__line" />
          </div>
        </div>
      ))}
    </div>
  );
}

export function SkeletonCards({ count = 3 }: { count?: number }) {
  return (
    <div className="skeleton-cards">
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton key={i} className="skeleton-card glass-card" />
      ))}
    </div>
  );
}
