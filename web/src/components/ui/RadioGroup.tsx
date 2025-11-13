'use client';

interface RadioOpt<T extends string> {
  value: T;
  label: string;
}

interface RadioGrpProps<T extends string> {
  name: string;
  options: RadioOpt<T>[];
  value: T;
  onChange: (val: T) => void;
}

export default function RadioGroup<T extends string>({ name, options, value, onChange }: RadioGrpProps<T>) {
  return (
    <div className="flex gap-4">
      {options.map((opt) => (
        <label key={opt.value} className="flex items-center cursor-pointer">
          <input
            type="radio"
            name={name}
            value={opt.value}
            checked={value === opt.value}
            onChange={(e) => onChange(e.target.value as T)}
            className="mr-2"
          />
          <span className="text-sm">{opt.label}</span>
        </label>
      ))}
    </div>
  );
}
