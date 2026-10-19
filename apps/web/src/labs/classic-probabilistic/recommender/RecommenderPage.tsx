import { useMemo, useState } from 'react';
import { Badge, Panel, Slider } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import {
  LabHeader,
  MetricsStrip,
  ParamPanel,
  StageNav,
  fix,
  type Metric,
  type ParamValue,
  type ParamValues,
} from '@/shared/lab-kit';
import {
  DEFAULT_RECOMMENDER,
  GENRES,
  ITEMS,
  USERS,
  runRecommender,
  type RecommenderRun,
  type RecommenderScenario,
} from './recommenderCore';
import {
  recommenderExperiments,
  recommenderParams,
  type RecommenderExperiment,
} from './recommenderSpec';

const toScenario = (p: ParamValues): RecommenderScenario => ({
  ...DEFAULT_RECOMMENDER,
  observed: Number(p.observed),
  factors: Number(p.factors),
  lambda: Number(p.lambda),
  method: p.method as RecommenderScenario['method'],
  iterations: Number(p.iterations),
  noise: Number(p.noise),
});

const fromScenario = (s: RecommenderScenario): ParamValues => ({
  observed: s.observed,
  factors: s.factors,
  lambda: s.lambda,
  method: s.method,
  iterations: s.iterations,
  noise: s.noise,
});

function Grid({
  values,
  known,
  label,
}: {
  values: number[][];
  known?: Set<string>;
  label: string;
}) {
  const palette = usePalette();
  const cell = 9;
  return (
    <svg
      width={ITEMS * cell}
      height={USERS * cell}
      viewBox={`0 0 ${ITEMS * cell} ${USERS * cell}`}
      className="max-w-full"
      role="img"
      aria-label={label}
    >
      {values.map((row, u) =>
        row.map((v, i) => {
          const unknown = known && !known.has(`${u},${i}`);
          return (
            <rect
              key={`${u}-${i}`}
              x={i * cell}
              y={u * cell}
              width={cell - 1}
              height={cell - 1}
              rx={1.5}
              fill={unknown ? palette.css.muted : v >= 3 ? palette.css.pos : palette.css.neg}
              opacity={unknown ? 0.12 : 0.15 + 0.85 * (Math.abs(v - 3) / 2)}
            />
          );
        }),
      )}
    </svg>
  );
}

function Tables({ run }: { run: RecommenderRun }) {
  const known = useMemo(() => new Set(run.train.map((r) => `${r.user},${r.item}`)), [run.train]);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start gap-x-10 gap-y-6">
        <figure className="space-y-2">
          <figcaption className="eyebrow">What we know</figcaption>
          <Grid
            values={run.world.stars}
            known={known}
            label="The ratings we know, people in rows and films in columns"
          />
        </figure>
        <figure className="space-y-2">
          <figcaption className="eyebrow">What the model predicts</figcaption>
          <Grid
            values={run.predicted}
            label="The model's predicted rating for every person and film"
          />
        </figure>
        <figure className="space-y-2">
          <figcaption className="eyebrow">What people would really say</figcaption>
          <Grid values={run.world.stars} label="The true ratings for every person and film" />
        </figure>
      </div>
      <p className="text-sm text-muted">
        Rows are people ({USERS}), columns are films ({ITEMS}).{' '}
        <span className="text-pos">Green</span> = liked (above 3 stars),{' '}
        <span className="text-neg">red</span> = disliked, faint grey = unknown. The model sees only
        the left panel and has to fill in the gaps; the right panel is the answer key. Films come in
        three genres, so a good model produces a visible block pattern.
      </p>
    </div>
  );
}

function Curves({ run }: { run: RecommenderRun }) {
  const palette = usePalette();
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <MiniChart
        title="Error (RMSE, in stars) after each iteration"
        x={run.trainRmse.map((_, i) => i)}
        series={[
          { name: 'known ratings', color: palette.css.brand, values: run.trainRmse },
          { name: 'unknown ratings', color: palette.css.neg, values: run.testRmse },
          {
            name: 'global average',
            color: palette.css.muted,
            values: run.testRmse.map(() => run.baselineMean),
            dashed: true,
          },
        ]}
        height={210}
      />
      <p className="text-sm text-muted">
        The blue curve is what the model is trained on; the red one is the error on ratings it never
        saw, which is what matters. When the two separate, the model is memorising. The dashed line
        is the error of always predicting the global average: a model has to get below it to be of
        any use.
      </p>
    </div>
  );
}

function ItemMap({ run }: { run: RecommenderRun }) {
  const palette = usePalette();
  const V = run.final.V;
  if (V.cols < 2)
    return (
      <p className="text-sm text-muted">
        Use at least two taste dimensions to draw the map of films.
      </p>
    );
  const xs = Array.from({ length: ITEMS }, (_, i) => V.get(i, 0));
  const ys = Array.from({ length: ITEMS }, (_, i) => V.get(i, 1));
  const w = 460;
  const h = 320;
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const sx = (v: number) => 24 + ((v - x0) / (x1 - x0 || 1)) * (w - 48);
  const sy = (v: number) => h - 24 - ((v - y0) / (y1 - y0 || 1)) * (h - 48);
  return (
    <div className="grid gap-8 lg:grid-cols-[auto_minmax(0,1fr)]">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        width={w}
        className="max-w-full rounded-xl border border-line"
        role="img"
        aria-label="Films placed by their first two learned taste factors, coloured by their true genre"
      >
        {xs.map((x, i) => (
          <circle
            key={i}
            cx={sx(x)}
            cy={sy(ys[i] as number)}
            r={6}
            fill={palette.css.classes[run.world.itemGenre[i]! % palette.css.classes.length]}
            opacity={0.8}
          />
        ))}
      </svg>
      <div className="space-y-2 text-sm text-muted" data-testid="map-note">
        <p>
          Each dot is a film placed by two of the numbers the model learned for it. Colours are the
          true genres, which the model never saw.
        </p>
        <p className="flex flex-wrap gap-3">
          {GENRES.map((g, i) => (
            <span key={g}>
              <span
                className="mr-1 inline-block size-2 rounded-full"
                style={{ background: palette.css.classes[i % palette.css.classes.length] }}
              />
              {g}
            </span>
          ))}
        </p>
        <p>
          Films of the same genre end up together because people who like one tend to like the
          others: the hidden factors rediscover the genres from ratings alone.
        </p>
      </div>
    </div>
  );
}

function ForYou({ run }: { run: RecommenderRun }) {
  const [user, setUser] = useState(0);
  const u = Math.min(user, USERS - 1);
  const known = new Set(run.train.filter((r) => r.user === u).map((r) => r.item));
  const picks = Array.from({ length: ITEMS }, (_, i) => i)
    .filter((i) => !known.has(i))
    .sort((a, b) => (run.predicted[u]![b] as number) - (run.predicted[u]![a] as number))
    .slice(0, 6);
  const liked = [...known]
    .filter((i) => run.world.stars[u]![i]! >= 4)
    .map((i) => `film ${i + 1} (${GENRES[run.world.itemGenre[i]!]})`);
  return (
    <div className="space-y-4">
      <Slider label="Person" min={0} max={USERS - 1} value={u} onValueChange={setUser} />
      <p className="text-sm text-muted">
        Rated {known.size} films;{' '}
        {liked.length
          ? `loved ${liked.slice(0, 4).join(', ')}${liked.length > 4 ? '…' : ''}.`
          : 'loved none of them yet.'}
      </p>
      <ol className="space-y-1.5" data-testid="picks">
        {picks.map((i) => (
          <li key={i} className="flex items-baseline gap-3 text-[0.85rem]">
            <span className="w-24 font-medium">Film {i + 1}</span>
            <span className="w-20 text-muted">{GENRES[run.world.itemGenre[i]!]}</span>
            <span className="font-mono text-pos">
              {fix(run.predicted[u]![i] as number, 1)} ★ predicted
            </span>
            <span className="font-mono text-faint">{run.world.stars[u]![i]} ★ real</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function RecommenderPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_RECOMMENDER));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const run = useMemo(() => runRecommender(scenario), [scenario]);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: RecommenderExperiment) => {
    setParams(fromScenario({ ...DEFAULT_RECOMMENDER, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = recommenderExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_RECOMMENDER, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const lastTest = run.testRmse[run.testRmse.length - 1] as number;
  const metrics: Metric[] = [
    {
      label: 'Error on unknown ratings',
      value: `${fix(lastTest, 2)} ★`,
      testId: 'test-rmse',
      hint: 'root-mean-square error in stars on ratings the model never saw',
    },
    {
      label: 'Error on known ratings',
      value: `${fix(run.trainRmse[run.trainRmse.length - 1], 2)} ★`,
    },
    { label: 'Always the global average', value: `${fix(run.baselineMean, 2)} ★` },
    { label: 'Average of each film', value: `${fix(run.baselineItem, 2)} ★` },
  ];

  const stages: [string, string][] = [
    ['lab-tables', 'The tables'],
    ['lab-curves', 'Learning'],
    ['lab-map', 'Films on a map'],
    ['lab-you', 'Recommendations'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Classic · probabilistic"
        title="Recommender systems"
        accent="filling in the missing ratings"
        blurb="Most people rate only a few films. Matrix factorisation explains the ratings they did give by a handful of hidden tastes and uses them to guess the rest, and to recommend what to watch next."
        badges={
          <>
            <Badge>{USERS} people</Badge>
            <Badge>{ITEMS} films</Badge>
            <Badge>{Math.round(scenario.observed * 100)}% rated</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={recommenderExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-tables" className="scroll-mt-24" title="The tables" eyebrow="01">
          <div className="space-y-6">
            <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
            <Tables run={run} />
          </div>
        </Panel>
        <Panel id="lab-curves" className="scroll-mt-24" title="Learning" eyebrow="02">
          <Curves run={run} />
        </Panel>
        <Panel id="lab-map" className="scroll-mt-24" title="Films on a map" eyebrow="03">
          <ItemMap run={run} />
        </Panel>
        <Panel id="lab-you" className="scroll-mt-24" title="Recommendations" eyebrow="04">
          <ForYou run={run} />
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="05">
          <div className="space-y-3 text-sm text-muted">
            <p>
              Write the table of ratings as R ≈ μ + U Vᵀ: every person gets a short vector of tastes
              (a row of U), every film a short vector of traits (a row of V), and a prediction is
              their dot product on top of the global average μ. Only the known entries enter the
              fit, and a penalty keeps the vectors small.
            </p>
            <p>
              <strong className="text-fg">Alternating least squares</strong> fixes the films and
              solves for each person exactly (a small ridge regression on the films they rated),
              then fixes the people and solves for each film; every pass can only lower the
              objective. <strong className="text-fg">Gradient descent</strong> instead nudges the
              vectors after each rating. Too few dimensions under-fit, too many with little penalty
              memorise: the error on unseen ratings is the judge.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="06">
          <ParamPanel specs={recommenderParams} values={params} onChange={setParam} columns={3} />
        </Panel>
      </div>
    </div>
  );
}
