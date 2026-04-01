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
      className={`rounded bg-accent/15 px-2.5 py-1 text-xs font-medium text-accent transition ${
        disabled ? 'cursor-not-allowed opacity-40' : 'hover:bg-accent/25'
      }`}
    >
      {label}
    </button>
  );
}
