import { AnimatePresence, motion } from 'framer-motion';
import { FlaskConical } from 'lucide-react';
import { cn } from '@/design-system';
import { LESSONS, type Lesson } from '../lessons';

/** One-click guided experiments, each one a prompt plus where to look and what to notice. */
export function LessonBar({
  activeId,
  onApply,
}: {
  activeId: string | null;
  onApply: (lesson: Lesson) => void;
}) {
  const active = LESSONS.find((l) => l.id === activeId);
  return (
    <div className="mb-6 rounded-2xl border border-line bg-surface/60 px-4 py-3 backdrop-blur-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1 flex items-center gap-1.5">
          <FlaskConical className="size-3.5" /> Experiments
        </span>
        {LESSONS.map((lesson) => (
          <button
            key={lesson.id}
            type="button"
            aria-pressed={lesson.id === activeId}
            onClick={() => onApply(lesson)}
            className={cn(
              'cursor-pointer rounded-full border px-3 py-1 text-[0.78rem] transition-all duration-200',
              lesson.id === activeId
                ? 'border-brand bg-brand text-brand-fg shadow-[0_0_0_3px_var(--brand-soft)]'
                : 'border-line-strong text-muted hover:border-brand hover:text-fg',
            )}
          >
            {lesson.title}
          </button>
        ))}
      </div>
      <AnimatePresence mode="wait" initial={false}>
        {active && (
          <motion.p
            key={active.id}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden pt-3 text-[0.88rem] leading-relaxed text-muted"
          >
            <span className="text-fg">What to look for: </span>
            {active.text}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
}
