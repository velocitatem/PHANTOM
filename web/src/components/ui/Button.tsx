import { ReactNode, ButtonHTMLAttributes } from 'react';

type BtnVariant = 'primary' | 'secondary';

interface BtnProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: BtnVariant;
  children: ReactNode;
  fullWidth?: boolean;
}

export default function Button({ variant = 'primary', children, fullWidth, className = '', ...props }: BtnProps) {
  const baseClass = variant === 'primary' ? 'btn-primary' : 'btn-secondary';
  const widthClass = fullWidth ? 'w-full' : '';

  return (
    <button className={`${baseClass} ${widthClass} ${className}`.trim()} {...props}>
      {children}
    </button>
  );
}
