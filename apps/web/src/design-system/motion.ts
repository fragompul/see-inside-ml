import type { Transition, Variants } from 'framer-motion';

/** Shared motion language: quick start, long soft landing. */
export const easeOutExpo = [0.16, 1, 0.3, 1] as const;

export const spring: Transition = { type: 'spring', stiffness: 380, damping: 32, mass: 0.8 };
export const springSoft: Transition = { type: 'spring', stiffness: 160, damping: 22 };

export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 18, filter: 'blur(6px)' },
  show: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: { duration: 0.7, ease: easeOutExpo },
  },
};

export const stagger = (delay = 0.07, start = 0): Variants => ({
  hidden: {},
  show: { transition: { staggerChildren: delay, delayChildren: start } },
});
