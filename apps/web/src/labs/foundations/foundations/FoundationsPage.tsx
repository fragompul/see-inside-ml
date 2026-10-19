import { useEffect, useState } from 'react';
import { Badge, Panel } from '@/design-system';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { ActivationsPanel } from './components/ActivationsPanel';
import { LandscapePanel } from './components/LandscapePanel';
import { LimitsPanel } from './components/LimitsPanel';
import { NeuronPanel } from './components/NeuronPanel';
import { LESSONS, type Lesson } from './lessons';
import { useFoundationsStore } from './store';

const STAGES = [
  ['fnd-neuron', 'A neuron'],
  ['fnd-activations', 'Activations'],
  ['fnd-descent', 'Gradient descent'],
  ['fnd-limits', 'Why layers'],
] as const;

export default function FoundationsPage() {
  const s = useFoundationsStore();
  const { descent, activationView } = s;
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    useFoundationsStore.getState().init();
    return () => useFoundationsStore.getState().setRunning(false);
  }, []);

  // An experiment stays "active" only while the lab still matches it.
  const active = LESSONS.find((l) => l.id === activeId);
  const stillActive =
    active &&
    (Object.entries(active.descent ?? {}) as [keyof typeof descent, unknown][]).every(
      ([key, value]) => descent[key] === value,
    ) &&
    (active.activation === undefined || active.activation === activationView);

  const apply = (lesson: Lesson) => {
    if (lesson.descent) s.patchDescent(lesson.descent);
    if (lesson.activation) s.setActivationView(lesson.activation);
    setActiveId(lesson.id);
    if (lesson.descent) {
      s.reset();
      if (lesson.run) s.setRunning(true);
    }
    window.setTimeout(
      () =>
        document
          .getElementById(lesson.scrollTo)
          ?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
      60,
    );
  };

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <header className="mb-8">
        <p className="eyebrow text-brand">01 · Lab</p>
        <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
          <span className="font-serif font-normal italic text-muted">Foundations:</span> neurons and
          learning
        </h1>
        <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">
          Start here. Every network in this lab is built from the same tiny part, the neuron, and
          learns the same way, by rolling downhill on a landscape of mistakes. Play with one neuron,
          see why activation functions matter and watch gradient descent find its way.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge>2 inputs</Badge>
          <Badge>1 neuron</Badge>
          <Badge>no setup needed</Badge>
        </div>
      </header>

      <nav
        aria-label="Sections of the lab"
        className="sticky top-16 z-30 -mx-4 mb-6 overflow-x-auto border-b border-line bg-bg/80 px-4 py-2.5 backdrop-blur-xl sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8"
      >
        <ol className="flex min-w-max items-center gap-1 font-mono text-[0.72rem]">
          {STAGES.map(([id, label], i) => (
            <li key={id} className="flex items-center gap-1">
              {i > 0 && <span className="text-faint">→</span>}
              <a
                href={`#${id}`}
                onClick={(e) => {
                  e.preventDefault();
                  document
                    .getElementById(id)
                    ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }}
                className="rounded-full px-3 py-1 text-muted transition-colors hover:bg-brand-soft hover:text-fg"
              >
                <span className="mr-1.5 text-brand">{String(i + 1).padStart(2, '0')}</span>
                {label}
              </a>
            </li>
          ))}
        </ol>
      </nav>

      <ExperimentBar lessons={LESSONS} activeId={stillActive ? activeId : null} onApply={apply} />

      <div className="flex flex-col gap-5">
        <Panel id="fnd-neuron" className="scroll-mt-24" title="One neuron, one line" eyebrow="01">
          <NeuronPanel />
        </Panel>
        <Panel
          id="fnd-activations"
          className="scroll-mt-24"
          title="Activation functions"
          eyebrow="02"
        >
          <ActivationsPanel />
        </Panel>
        <Panel
          id="fnd-descent"
          className="scroll-mt-24"
          title="Learning is rolling downhill"
          eyebrow="03"
        >
          <LandscapePanel />
        </Panel>
        <Panel
          id="fnd-limits"
          className="scroll-mt-24"
          title="What one neuron cannot do"
          eyebrow="04"
        >
          <LimitsPanel />
        </Panel>
      </div>
    </div>
  );
}
