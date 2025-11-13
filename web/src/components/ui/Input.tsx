import { InputHTMLAttributes, ReactNode } from 'react';

interface InpProps extends InputHTMLAttributes<HTMLInputElement> {
    icon?: ReactNode;
}

export default function Input({ icon, className = '', style, ...props }: InpProps) {
    const padClass = icon ? 'pl-10' : '';
    // Fallback if a custom CSS rule still overrides Tailwind
    const mergedStyle = icon ? { paddingInlineStart: '2.5rem', ...style } : style;

    return (
        <div className="relative">
            {icon && (
                <div
                    aria-hidden
                    className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-gray-400 z-10"
                >
                    {icon}
                </div>
            )}
            <input
                className={`input-field ${className} ${padClass}`}
                style={mergedStyle}
                {...props}
            />
        </div>
    );
}
