import { motion } from 'framer-motion';
import type { ReactNode } from 'react';
import { ArrowDown, ArrowRight, ArrowUpRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Badge, Button, Card, cn, fadeUp, stagger } from '@/design-system';
import { HeroField } from './HeroField';
import { REPO_URL } from './Layout';
import { activeTracks, labsOf, type ModuleAccent, type ModuleDefinition } from './registry';

const accentText: Record<ModuleAccent, string> = {
  pos: 'text-pos',
  neg: 'text-neg',
  brand: 'text-brand',
  ok: 'text-ok',
};

const accentBar: Record<ModuleAccent, string> = {
  pos: 'bg-pos',
  neg: 'bg-neg',
  brand: 'bg-brand',
  ok: 'bg-ok',
};

function Hero() {
  return (
    <section className="relative isolate overflow-hidden border-b border-line">
      <div aria-hidden className="bg-dots absolute inset-0 -z-20" />
      <div
        aria-hidden
        className="absolute -top-40 -left-32 -z-10 size-[34rem] rounded-full bg-brand/20 blur-[120px] [animation:drift_18s_ease-in-out_infinite]"
      />
      <div
        aria-hidden
        className="absolute -right-40 bottom-[-12rem] -z-10 size-[30rem] rounded-full bg-pos/15 blur-[120px] [animation:drift_22s_ease-in-out_infinite_reverse]"
      />
      <HeroField className="absolute inset-0 -z-10 size-full opacity-35 sm:opacity-50 lg:opacity-100 lg:[mask-image:linear-gradient(to_right,transparent_28%,#000_52%)]" />

      <div className="mx-auto flex min-h-[calc(100svh-4rem)] max-w-7xl flex-col justify-center px-5 py-24 sm:px-8">
        <motion.div
          variants={stagger(0.09, 0.1)}
          initial="hidden"
          animate="show"
          className="max-w-2xl"
        >
          <motion.div variants={fadeUp} className="mb-7 flex flex-wrap items-center gap-2">
            <Badge tone="brand">Open source</Badge>
            <Badge>Runs in your browser</Badge>
            <Badge>Built from scratch</Badge>
          </motion.div>

          <motion.h1
            variants={fadeUp}
            className="font-display text-[clamp(2.9rem,8vw,6.2rem)] leading-[0.95] font-semibold tracking-[-0.04em]"
          >
            See inside
            <br />
            the{' '}
            <em className="text-gradient-signal pr-[0.08em] font-serif font-normal tracking-[-0.02em] italic">
              machine.
            </em>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            className="mt-7 max-w-xl text-[1.08rem] leading-relaxed text-muted"
          >
            An interactive lab where every neuron, weight and gradient is something you can grab,
            tweak and watch. From a single perceptron to transformers and diffusion, nothing is a
            black box.
          </motion.p>

          <motion.div variants={fadeUp} className="mt-9 flex flex-wrap items-center gap-3">
            <Link to="/mlp" className="contents">
              <Button size="lg" tabIndex={-1}>
                Open the playground <ArrowRight className="size-4" />
              </Button>
            </Link>
            <Button
              size="lg"
              variant="secondary"
              onClick={() => document.getElementById('path')?.scrollIntoView()}
            >
              Learning path <ArrowDown className="size-4" />
            </Button>
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="contents">
              <Button size="lg" variant="ghost" tabIndex={-1}>
                Star on GitHub <ArrowUpRight className="size-4" />
              </Button>
            </a>
          </motion.div>

          <motion.dl
            variants={fadeUp}
            className="mt-14 flex flex-wrap gap-x-7 gap-y-3 font-mono text-[0.72rem] tracking-wide text-muted"
          >
            {[
              ['bg-pos', 'positive weight'],
              ['bg-neg', 'negative weight'],
              ['bg-brand', 'backpropagation'],
            ].map(([dot, label]) => (
              <div key={label} className="flex items-center gap-2">
                <dt className={cn('size-2 rounded-full', dot)} aria-hidden />
                <dd>{label}</dd>
              </div>
            ))}
          </motion.dl>
        </motion.div>
      </div>
    </section>
  );
}

function statusBadge(status: ModuleDefinition['status']) {
  if (status === 'live') return <Badge tone="ok">Live</Badge>;
  if (status === 'building') return <Badge tone="pos">Building now</Badge>;
  return <Badge>Coming soon</Badge>;
}

function CardLink({
  path,
  title,
  children,
}: {
  path?: string;
  title: string;
  children: ReactNode;
}) {
  if (!path) return <>{children}</>;
  return (
    <Link to={path} aria-label={`Open ${title}`} className="block h-full rounded-2xl">
      {children}
    </Link>
  );
}

function ModuleCard({ module, index }: { module: ModuleDefinition; index: number }) {
  return (
    <motion.li variants={fadeUp} className="list-none">
      <CardLink path={module.path} title={module.title}>
        <Card className="h-full">
          <div
            className={cn('absolute inset-x-0 top-0 h-px opacity-70', accentBar[module.accent])}
          />
          <div className="flex h-full flex-col p-6">
            <div className="flex items-start justify-between">
              <span className={cn('font-mono text-sm tabular-nums', accentText[module.accent])}>
                {String(index + 1).padStart(2, '0')}
              </span>
              {statusBadge(module.status)}
            </div>
            <h3 className="mt-6 font-display text-[1.45rem] leading-tight font-semibold tracking-tight">
              {module.title}
            </h3>
            <p className="mt-2.5 text-[0.92rem] leading-relaxed text-muted">{module.tagline}</p>
            <ul className="mt-5 flex flex-wrap gap-1.5" aria-label="Topics">
              {module.topics.map((topic) => (
                <li
                  key={topic}
                  className="rounded-md border border-line bg-bg-soft px-2 py-1 font-mono text-[0.68rem] text-muted"
                >
                  {topic}
                </li>
              ))}
            </ul>
            <div className="mt-auto flex items-center justify-between pt-5">
              <p className="eyebrow">{module.level}</p>
              {module.path && (
                <span className="flex items-center gap-1.5 font-mono text-[0.72rem] text-brand">
                  Open lab{' '}
                  <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" />
                </span>
              )}
            </div>
          </div>
        </Card>
      </CardLink>
    </motion.li>
  );
}

function LearningPath() {
  return (
    <section id="path" className="mx-auto max-w-7xl scroll-mt-20 px-5 py-28 sm:px-8">
      <motion.div
        variants={stagger(0.06)}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: '-80px' }}
      >
        <motion.p variants={fadeUp} className="eyebrow">
          The learning path
        </motion.p>
        <motion.h2
          variants={fadeUp}
          className="mt-4 max-w-3xl font-display text-[clamp(2rem,4.5vw,3.4rem)] leading-[1.05] font-semibold tracking-[-0.03em]"
        >
          One idea at a time, from a straight line to a{' '}
          <em className="font-serif font-normal italic">language model.</em>
        </motion.h2>
        <motion.p variants={fadeUp} className="mt-5 max-w-2xl text-muted">
          Labs are grouped in tracks and ordered so each one builds on the last. Open any card to
          start: every lab runs entirely in your browser, with nothing to install or download.
        </motion.p>

        <div className="mt-14 space-y-16">
          {activeTracks().map((track) => (
            <div key={track.id} aria-labelledby={`track-${track.id}`}>
              <div className="mb-6 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-line pb-3">
                <h3
                  id={`track-${track.id}`}
                  className="font-display text-xl font-semibold tracking-tight"
                >
                  {track.title}
                </h3>
                <p className="max-w-xl text-[0.86rem] text-muted">{track.tagline}</p>
              </div>
              <ol className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
                {labsOf(track.id).map((module, index) => (
                  <ModuleCard key={module.id} module={module} index={index} />
                ))}
              </ol>
            </div>
          ))}
        </div>
      </motion.div>
    </section>
  );
}

export function Home() {
  return (
    <>
      <Hero />
      <LearningPath />
    </>
  );
}
