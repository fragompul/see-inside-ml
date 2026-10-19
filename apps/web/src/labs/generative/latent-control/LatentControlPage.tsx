import { useMemo, useState } from 'react';
import { IMAGE_SIZE } from '@sim/deep';
import { Badge, Panel } from '@/design-system';
import { usePalette } from '@/shared/canvas/colors';
import { useCompute } from '@/shared/compute/useCompute';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
import { MiniChart } from '@/shared/components/MiniChart';
import { PixelCanvas } from '@/shared/components/PixelCanvas';
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
  DEFAULT_CONTROL,
  type ControlScenario,
  type ConditionalResult,
  type VqResult,
} from './controlCore';
import { controlExperiments, controlParams, type ControlExperiment } from './controlSpec';

const toScenario = (p: ParamValues): ControlScenario => ({
  ...DEFAULT_CONTROL,
  mode: p.mode as ControlScenario['mode'],
  conditioned: Boolean(p.conditioned),
  beta: Number(p.beta),
  entries: Number(p.entries),
  revive: Boolean(p.revive),
});

const fromScenario = (s: ControlScenario): ParamValues => ({
  mode: s.mode,
  conditioned: s.conditioned,
  beta: s.beta,
  entries: s.entries,
  revive: s.revive,
});

const pct = (v: number) => `${fix(v * 100, 0)}%`;

function ConditionalView({ r }: { r: ConditionalResult }) {
  const metrics: Metric[] = [
    {
      label: 'Right digit when asked for one',
      value: pct(r.control),
      testId: 'control',
      hint: 'read by an independent classifier',
    },
    { label: 'Reconstruction loss', value: fix(r.testRecon, 1), hint: 'new digits' },
    { label: 'Information in the code (KL)', value: fix(r.testKl, 1) },
  ];
  return (
    <Panel id="lab-output" className="scroll-mt-24" title="What it draws" eyebrow="01">
      <div className="space-y-6">
        <MetricsStrip metrics={metrics} className="lg:grid-cols-3" />
        <div className="overflow-x-auto">
          <div className="inline-grid gap-1" style={{ gridTemplateColumns: 'auto repeat(8, 1fr)' }}>
            {r.grid.map((row, digit) => (
              <div key={digit} className="contents">
                <span className="self-center pr-2 font-mono text-xs text-muted">
                  {r.scenario.conditioned ? `asked for ${digit}` : 'class not given'}
                </span>
                {row.map((img, i) => (
                  <PixelCanvas
                    key={i}
                    values={img}
                    side={IMAGE_SIZE}
                    size={44}
                    ariaLabel={`Image for code ${i + 1}, row ${digit}`}
                    className="rounded"
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
        <p className="text-sm text-muted">
          Each column is one fixed code; each row asks the decoder for a different digit. When the
          model has been given the class, a column keeps its style (slant, thickness) across the
          rows while the digit changes. When it has not, every row is the same picture, because the
          decoder has nothing to change.
        </p>
      </div>
    </Panel>
  );
}

function VqView({ r }: { r: VqResult }) {
  const palette = usePalette();
  const last = r.testRecon[r.testRecon.length - 1] as number;
  const metrics: Metric[] = [
    {
      label: 'Entries never used',
      value: `${r.deadEntries} of ${r.usage.length}`,
      testId: 'dead-entries',
    },
    { label: 'Reconstruction loss', value: fix(last, 1), hint: 'new digits' },
    { label: 'Purity of an entry', value: pct(r.purity), hint: 'share of its most common digit' },
    {
      label: 'Prototype read as that digit',
      value: pct(r.prototypeAgreement),
      hint: 'by an independent classifier',
    },
  ];
  return (
    <>
      <Panel id="lab-output" className="scroll-mt-24" title="The codebook" eyebrow="01">
        <div className="space-y-6">
          <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
          <div className="flex flex-wrap gap-3">
            {r.prototypes.map((img, k) => {
              const m = r.majority[k] as { digit: number; purity: number };
              const used = r.usage[k] as number;
              return (
                <figure key={k} className="text-center">
                  <PixelCanvas
                    values={img}
                    side={IMAGE_SIZE}
                    size={64}
                    ariaLabel={`Prototype of codebook entry ${k + 1}`}
                    className={used === 0 ? 'rounded-lg opacity-30' : 'rounded-lg'}
                  />
                  <figcaption className="mt-1 font-mono text-[11px] text-muted">
                    {used === 0 ? 'unused' : `${used} images · mostly ${m.digit}`}
                  </figcaption>
                </figure>
              );
            })}
          </div>
          <p className="text-sm text-muted">
            Each picture is what the decoder paints from one codebook entry. Faded ones were never
            chosen by any training image. The caption says how many training images snap to the
            entry and which digit most of them are.
          </p>
        </div>
      </Panel>
      <Panel id="lab-curve" className="scroll-mt-24" title="Learning" eyebrow="02">
        <MiniChart
          title="Reconstruction loss on new digits by epoch"
          x={r.testRecon.map((_, i) => i + 1)}
          xLabel="epoch"
          series={[{ name: 'loss', color: palette.css.brand, values: r.testRecon }]}
          height={200}
        />
      </Panel>
    </>
  );
}

export default function LatentControlPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_CONTROL));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const { result, error } = useCompute('latent-control', scenario);
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  const apply = (exp: ControlExperiment) => {
    setParams(fromScenario({ ...DEFAULT_CONTROL, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = controlExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_CONTROL, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const stages: [string, string][] = [
    ['lab-output', scenario.mode === 'conditional' ? 'What it draws' : 'The codebook'],
    ...(scenario.mode === 'vq' ? ([['lab-curve', 'Learning']] as [string, string][]) : []),
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Generative models"
        title="Steering a VAE: conditional and discrete codes"
        accent="choosing what it makes"
        blurb="A plain VAE draws something from its code, with no say in what. Tell it the digit and the code is freed for style. Or replace the continuous code by a pick from a small codebook, and each entry becomes a prototype. Two ways of putting structure into a latent space."
        badges={
          <>
            <Badge>{scenario.mode === 'conditional' ? 'Conditional VAE' : 'VQ-VAE'}</Badge>
            <Badge>25 epochs of training</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={controlExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        {result ? (
          result.kind === 'conditional' ? (
            <ConditionalView r={result} />
          ) : (
            <VqView r={result} />
          )
        ) : (
          <Panel
            id="lab-output"
            className="scroll-mt-24"
            title={scenario.mode === 'conditional' ? 'What it draws' : 'The codebook'}
            eyebrow="01"
          >
            <p className="text-sm text-muted" role="status">
              {error
                ? `Something went wrong: ${error}`
                : 'Training the model and the classifier that judges it…'}
            </p>
          </Panel>
        )}
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              <strong className="text-fg">Conditional VAE.</strong> The digit class, as a one-hot
              vector, is attached to the input of the encoder and to the code before it enters the
              decoder. The usual VAE loss (reconstruction plus β times the KL term) is unchanged.
              Because the decoder is handed the class directly, the code does not need to carry it,
              and asking for a different class changes the digit while the code keeps the style.
            </p>
            <p>
              <strong className="text-fg">VQ-VAE.</strong> The encoder output is replaced by the
              nearest of K learned vectors. That step has no gradient, so the decoder&rsquo;s
              gradient is copied straight to the encoder; the codebook entries are pulled towards
              the codes assigned to them and the encoder is held close to its entry. Entries that no
              image ever picks stop learning, so here they are optionally moved onto the codes of
              random images. The tests check the gradients of both models against finite
              differences.
            </p>
          </div>
        </Panel>
        <Panel
          id="lab-controls"
          className="scroll-mt-24"
          title="Settings"
          eyebrow={scenario.mode === 'vq' ? '04' : '02'}
        >
          <ParamPanel specs={controlParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
