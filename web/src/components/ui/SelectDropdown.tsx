'use client';

import { useState, useRef, useEffect, ReactNode } from 'react';

export interface SelectOption {
  value: string;
  label: string;
  sublabel?: string;
}

interface SelectDropdownProps {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  icon?: ReactNode;
  required?: boolean;
  id?: string;
}

export default function SelectDropdown({
  value,
  onChange,
  options,
  placeholder = 'Select...',
  icon,
  required,
  id,
}: SelectDropdownProps) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setFilter('');
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const selectedOption = options.find((o) => o.value === value);
  const filtered = options.filter(
    (o) =>
      o.label.toLowerCase().includes(filter.toLowerCase()) ||
      o.value.toLowerCase().includes(filter.toLowerCase()) ||
      o.sublabel?.toLowerCase().includes(filter.toLowerCase())
  );

  const handleSelect = (opt: SelectOption) => {
    onChange(opt.value);
    setOpen(false);
    setFilter('');
  };

  return (
    <div className="relative" ref={ref}>
      <div
        className="input-field flex items-center gap-2 cursor-pointer box-border"
        onClick={() => {
          setOpen(true);
          setTimeout(() => inputRef.current?.focus(), 0);
        }}
      >
        {icon && <span className="text-[var(--text-secondary)]">{icon}</span>}
        {open ? (
          <input
            ref={inputRef}
            type="text"
            id={id}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder={placeholder}
            className="flex-1 bg-transparent outline-none text-sm text-[var(--text-primary)]"
          />
        ) : (
          <span className={`flex-1 text-sm ${value ? 'text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>
            {selectedOption ? selectedOption.label : placeholder}
          </span>
        )}
        <svg
          className={`w-4 h-4 text-[var(--text-secondary)] transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </div>
      {open && (
        <div className="absolute z-20 mt-1 w-full bg-[var(--bg-primary)] border-2 border-[var(--accent-primary)] rounded-md shadow-lg max-h-60 overflow-y-auto">
          {filtered.length === 0 ? (
            <div className="px-4 py-3 text-sm text-[var(--text-secondary)]">No results</div>
          ) : (
            filtered.map((opt) => (
              <div
                key={opt.value}
                onClick={() => handleSelect(opt)}
                className={`px-4 py-2 cursor-pointer transition-colors hover:bg-[var(--accent-primary-light)] ${
                  opt.value === value ? 'bg-[var(--accent-primary-light)]' : ''
                }`}
              >
                <div className="text-sm font-medium text-[var(--text-primary)]">{opt.label}</div>
                {opt.sublabel && <div className="text-xs text-[var(--text-secondary)]">{opt.sublabel}</div>}
              </div>
            ))
          )}
        </div>
      )}
      {required && !value && (
        <input type="text" required className="sr-only" tabIndex={-1} value="" onChange={() => {}} />
      )}
    </div>
  );
}
