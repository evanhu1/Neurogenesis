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
  primary: 'bg-accent font-semibold text-white shadow-sm hover:bg-green-800',
  ghost: 'border border-line bg-panel text-ink/75 shadow-sm hover:bg-surface/70 hover:text-ink',
  ghostActive: 'border border-accent/40 bg-accent/10 text-accent hover:bg-accent/15',
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
