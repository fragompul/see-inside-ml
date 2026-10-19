import { useEffect, useMemo, useState } from 'react';
import { createRng } from '@sim/core';
import { DEFAULT_STYLE, IMAGE_SIZE, augmentImage, renderDigit } from '@sim/deep';
import { Badge, Panel } from '@/design-system';
import { compute } from '@/shared/compute/client';
import { ExperimentBar } from '@/shared/components/ExperimentBar';
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
  AUGMENTATIONS,
  DEFAULT_AUGMENT,
  SEEDS,
  type AugmentId,
  type AugmentRun,
  type AugmentScenario,
} from './augmentCore';
import { augmentExperiments, augmentParams, type AugmentExperiment } from './augmentSpec';

const toScenario = (p: ParamValues): AugmentScenario => ({
  ...DEFAULT_AUGMENT,
  augmentation: p.augmentation as AugmentId,
  perClass: Number(p.perClass),
  copies: Number(p.copies),
});

const fromScenario = (s: AugmentScenario): ParamValues => ({
  augmentation: s.augmentation,
  perClass: s.perClass,
  copies: s.copies,
});

const pct = (v: number) => `${fix(v * 100, 0)}%`;
const IDS = Object.keys(AUGMENTATIONS) as AugmentId[];

export default function AugmentationPage() {
  const [params, setParams] = useState<ParamValues>(() => fromScenario(DEFAULT_AUGMENT));
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = useMemo(() => toScenario(params), [params]);
  const scenarioKey = JSON.stringify(scenario);
  const [store, setStore] = useState<{ key: string; byId: Partial<Record<AugmentId, AugmentRun>> }>(
    {
      key: '',
      byId: {},
    },
  );
  const results = store.key === scenarioKey ? store.byId : {};
  const setParam = (key: string, value: ParamValue) => setParams((p) => ({ ...p, [key]: value }));

  // the chosen augmentation first, then the others for the comparison table, one after another in the worker
  useEffect(() => {
    let cancelled = false;
    const order: AugmentId[] = [
      scenario.augmentation,
      ...IDS.filter((id) => id !== scenario.augmentation),
    ];
    void (async () => {
      for (const id of order) {
        if (cancelled) return;
        const run = await compute('augment', { ...scenario, augmentation: id }).catch(() => null);
        if (cancelled || !run) return;
        setStore((prev) => ({
          key: scenarioKey,
          byId: { ...(prev.key === scenarioKey ? prev.byId : {}), [id]: run },
        }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [scenario, scenarioKey]);

  const apply = (exp: AugmentExperiment) => {
    setParams(fromScenario({ ...DEFAULT_AUGMENT, ...exp.scenario }));
    setActiveId(exp.id);
  };
  const stillActive = (() => {
    const exp = augmentExperiments.find((e) => e.id === activeId);
    return (
      !!exp &&
      JSON.stringify(fromScenario({ ...DEFAULT_AUGMENT, ...exp.scenario })) ===
        JSON.stringify(params)
    );
  })();

  const shown = results[scenario.augmentation];
  const metrics: Metric[] = [
    {
      label: 'Accuracy on new digits',
      value: shown ? pct(shown.clean) : '…',
      testId: 'clean-accuracy',
      hint: `mean of ${SEEDS} runs`,
    },
    {
      label: 'Shifted, tilted and noisy',
      value: shown ? pct(shown.wild) : '…',
      hint: 'new digits seen through a worse camera',
    },
    { label: 'On the training pictures', value: shown ? pct(shown.train) : '…' },
    { label: 'Training set size', value: shown ? String(shown.examples) : '…' },
  ];

  const gallery = useMemo(() => {
    const settings = (
      AUGMENTATIONS[scenario.augmentation] as { settings: Parameters<typeof augmentImage>[1] }
    ).settings;
    const rng = createRng(77);
    return [2, 5, 7].map((digit) => {
      const original = renderDigit(digit, createRng(digit * 13 + 1), DEFAULT_STYLE);
      return {
        digit,
        pictures: [
          original,
          ...Array.from({ length: 5 }, () => augmentImage(original, settings, rng)),
        ],
      };
    });
  }, [scenario.augmentation]);

  const stages: [string, string][] = [
    ['lab-results', 'Does it help?'],
    ['lab-copies', 'The copies'],
    ['lab-how', 'How it works'],
    ['lab-controls', 'Settings'],
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <LabHeader
        eyebrow="Deep learning"
        title="Data augmentation"
        accent="more data from the same data"
        blurb="When labelled examples are scarce, make more by changing the ones you have in ways that cannot change the label: shift, tilt, add noise. See when this works, and why the copies have to resemble the changes you will actually meet."
        badges={
          <>
            <Badge>{(AUGMENTATIONS[scenario.augmentation] as { label: string }).label}</Badge>
            <Badge>{scenario.perClass} real examples per digit</Badge>
          </>
        }
      />
      <StageNav stages={stages} />
      <ExperimentBar
        lessons={augmentExperiments}
        activeId={stillActive ? activeId : null}
        onApply={apply}
      />

      <div className="flex flex-col gap-5">
        <Panel id="lab-results" className="scroll-mt-24" title="Does it help?" eyebrow="01">
          <div className="space-y-6" aria-busy={!shown}>
            <MetricsStrip metrics={metrics} className="lg:grid-cols-4" />
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-muted">
                  <tr>
                    <th className="py-2 pr-4 font-medium">Augmentation</th>
                    <th className="py-2 pr-4 font-medium">New digits</th>
                    <th className="py-2 font-medium">Shifted, tilted, noisy</th>
                  </tr>
                </thead>
                <tbody>
                  {IDS.map((id) => {
                    const r = results[id];
                    return (
                      <tr key={id} className="border-t border-line">
                        <td className="py-2 pr-4">
                          {(AUGMENTATIONS[id] as { label: string }).label}
                          {id === scenario.augmentation ? ' (shown above)' : ''}
                        </td>
                        <td className="py-2 pr-4 font-mono">{r ? pct(r.clean) : '…'}</td>
                        <td className="py-2 font-mono">{r ? pct(r.wild) : '…'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </Panel>
        <Panel id="lab-copies" className="scroll-mt-24" title="The copies" eyebrow="02">
          <div className="space-y-4">
            {gallery.map((row) => (
              <div key={row.digit} className="flex flex-wrap items-center gap-2">
                {row.pictures.map((p, i) => (
                  <PixelCanvas
                    key={i}
                    values={p}
                    side={IMAGE_SIZE}
                    size={72}
                    ariaLabel={
                      i === 0
                        ? `Original digit ${row.digit}`
                        : `Augmented copy ${i} of digit ${row.digit}`
                    }
                    className={i === 0 ? 'rounded-lg ring-2 ring-brand' : 'rounded-lg'}
                  />
                ))}
              </div>
            ))}
            <p className="text-sm text-muted">
              The picture with the ring is the original; the others are random copies made with the
              chosen augmentation. Every copy keeps its label.
            </p>
          </div>
        </Panel>
        <Panel id="lab-how" className="scroll-mt-24" title="How it works" eyebrow="03">
          <div className="space-y-3 text-sm text-muted">
            <p>
              A network with a few hundred weights per digit and only a handful of examples can
              match every training picture and still fail on the next one. Each augmented copy is a
              new training example with the same label, so the network is pushed to give the same
              answer for a picture and for its shifted, tilted or noisy versions.
            </p>
            <p>
              The &ldquo;wild&rdquo; test pictures are new digits with a random shift of up to two
              pixels, a tilt of up to 15° and noise. The shift, rotation and noise functions are
              tested exactly: a shift moves the centre of mass by the shift, a zero rotation is the
              identity, and the copies stay inside their limits. Accuracies are means over three
              runs with different random digits.
            </p>
          </div>
        </Panel>
        <Panel id="lab-controls" className="scroll-mt-24" title="Settings" eyebrow="04">
          <ParamPanel specs={augmentParams} values={params} onChange={setParam} columns={2} />
        </Panel>
      </div>
    </div>
  );
}
