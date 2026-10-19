import { AnimatePresence, motion } from 'framer-motion';
import { FlaskConical } from 'lucide-react';
import { cn } from '@/design-system';

export interface ExperimentLike {
  id: string;
  title: string;
  /** What to look for once the experiment runs. */
  text: string;
}

/** One-click experiments: each sets up a lab and says what to look for. */
export function ExperimentBar<L extends ExperimentLike>({
  lessons,
  activeId,
  onApply,
}: {
  lessons: readonly L[];
  activeId: string | null;
  onApply: (lesson: L) => void;
}) {
  const active = lessons.find((l) => l.id === activeId);
  return (
    <div className="mb-6 rounded-2xl border border-line bg-surface/60 px-4 py-3 backdrop-blur-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1 flex items-center gap-1.5">
          <FlaskConical className="size-3.5" /> Experiments
        </span>
        {lessons.map((lesson) => (
          <button
            key={lesson.id}
            type="button"
            aria-pressed={lesson.id === activeId}
            data-experiment={lesson.id}
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
