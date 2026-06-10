import type { ReactNode } from 'react';

type ControlButtonProps = {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'ghost';
  active?: boolean;
  icon?: ReactNode;
  className?: string;
};

const VARIANT_CLASSES = {
  primary: 'bg-accent font-semibold text-slate-950 hover:bg-sky-300',
  ghost: 'border border-white/10 bg-surface/60 text-ink/75 hover:bg-surface hover:text-ink',
  ghostActive: 'border border-accent/50 bg-accent/15 text-accent hover:bg-accent/25',
};

export function ControlButton({
  label,
  onClick,
  disabled = false,
  variant = 'ghost',
  active = false,
  icon,
  className = '',
}: ControlButtonProps) {
  const variantClass =
    variant === 'primary'
      ? VARIANT_CLASSES.primary
      : active
        ? VARIANT_CLASSES.ghostActive
        : VARIANT_CLASSES.ghost;
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center justify-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs transition ${variantClass} ${
        disabled ? 'cursor-not-allowed opacity-40' : ''
      } ${className}`}
    >
      {icon}
      {label}
    </button>
  );
}
