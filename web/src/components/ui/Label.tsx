import { ReactNode, LabelHTMLAttributes } from 'react';

interface LblProps extends LabelHTMLAttributes<HTMLLabelElement> {
  children: ReactNode;
}

export default function Label({ children, className = '', ...props }: LblProps) {
  return (
    <label className={`block text-sm font-medium mb-2 ${className}`.trim()} {...props}>
      {children}
    </label>
  );
}
