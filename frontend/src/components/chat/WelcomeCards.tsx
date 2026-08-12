import { BookOpen, Brain, Info, Sparkles, Upload } from 'lucide-react';
import { modeCardTooltips } from '../../data/mockData';
import { Tooltip } from '../ui/Tooltip';

export function WelcomeCards({
  onSelectMode,
  onUpload,
}: {
  onSelectMode: (mode: 'direct' | 'catalog' | 'ai') => void;
  onUpload: () => void;
}) {
  const cards = [
    {
      id: 'direct' as const,
      icon: <Sparkles size={20} />,
      color: 'orange',
      title: 'Direct',
      desc: 'Get precise answers from your document',
      tip: modeCardTooltips.direct,
    },
    {
      id: 'catalog' as const,
      icon: <BookOpen size={20} />,
      color: 'blue',
      title: 'Catalog',
      desc: 'Browse the knowledge catalog',
      tip: modeCardTooltips.catalog,
    },
    {
      id: 'ai' as const,
      icon: <Brain size={20} />,
      color: 'green',
      title: 'AI Answer',
      desc: 'Let the AI synthesize a full response',
      tip: modeCardTooltips.ai,
    },
    {
      id: 'upload' as const,
      icon: <Upload size={20} />,
      color: 'pink',
      title: 'Upload a file',
      desc: 'Add a new document and start chatting',
      tip: modeCardTooltips.upload,
    },
  ];

  return (
    <div className="welcome-cards">
      {cards.map((card) => (
        <Tooltip key={card.id} content={card.tip}>
          <button
            className={`welcome-card glass-card welcome-card--${card.color}`}
            onClick={() =>
              card.id === 'upload' ? onUpload() : onSelectMode(card.id)
            }
            type="button"
          >
            <span className={`welcome-card__icon welcome-card__icon--${card.color}`}>
              {card.icon}
            </span>
            <span className="welcome-card__text">
              <strong>
                {card.title}
                <Info size={12} className="welcome-card__info" aria-hidden="true" />
              </strong>
              <small>{card.desc}</small>
            </span>
            <span className="welcome-card__plus">+</span>
          </button>
        </Tooltip>
      ))}
    </div>
  );
}
