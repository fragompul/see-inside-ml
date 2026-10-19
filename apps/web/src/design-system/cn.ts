import clsx, { type ClassValue } from 'clsx';

/** Tiny class-name combiner used by every component. */
export const cn = (...inputs: ClassValue[]) => clsx(inputs);
