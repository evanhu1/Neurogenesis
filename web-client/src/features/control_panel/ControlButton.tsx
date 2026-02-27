type ControlButtonProps = {
  label: string;
  onClick: () => void;
  disabled?: boolean;
};

export function ControlButton({ label, onClick, disabled = false }: ControlButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-white transition ${
        disabled ? 'cursor-not-allowed opacity-50 grayscale' : 'hover:brightness-110'
      }`}
    >
      {label}
    </button>
  );
}
