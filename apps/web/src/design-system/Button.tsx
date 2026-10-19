import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { cn } from './cn';

type Variant = 'primary' | 'secondary' | 'ghost';
type Size = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const variants: Record<Variant, string> = {
  primary:
    'bg-fg text-bg hover:bg-brand hover:text-brand-fg shadow-[0_8px_24px_-10px_var(--brand)]',
  secondary: 'border border-line-strong bg-raised/60 text-fg backdrop-blur hover:border-brand',
  ghost: 'text-muted hover:text-fg hover:bg-brand-soft',
};

const sizes: Record<Size, string> = {
  sm: 'h-8 px-3 text-[0.8rem] gap-1.5',
  md: 'h-10 px-4 text-sm gap-2',
  lg: 'h-12 px-6 text-[0.95rem] gap-2.5',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'primary', size = 'md', className, type = 'button', ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        'inline-flex cursor-pointer items-center justify-center rounded-full font-medium tracking-tight',
        'transition-[background,color,border-color,transform,box-shadow] duration-300 ease-out',
        'active:scale-[0.97] disabled:pointer-events-none disabled:opacity-45',
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  );
});
