import { InputHTMLAttributes } from 'react';

interface DateInpProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {}

export default function DateInput({ className = '', ...props }: DateInpProps) {
  return <input type="date" className={`input-field ${className}`.trim()} {...props} />;
}
