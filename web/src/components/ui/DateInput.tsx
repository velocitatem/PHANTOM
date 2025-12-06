import { InputHTMLAttributes, useMemo } from 'react';

interface DateInpProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {}

export default function DateInput({ className = '', ...props }: DateInpProps) {
  const { minDate, maxDate } = useMemo(() => {
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);

    const tenDaysOut = new Date(tomorrow);
    tenDaysOut.setDate(tomorrow.getDate() + 9); // tomorrow + 9 = 10 days total

    return {
      minDate: tomorrow.toISOString().split('T')[0],
      maxDate: tenDaysOut.toISOString().split('T')[0]
    };
  }, []);

  return (
    <input
      type="date"
      className={`input-field ${className}`.trim()}
      min={minDate}
      max={maxDate}
      {...props}
    />
  );
}
