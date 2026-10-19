import { AnimatePresence } from 'framer-motion';
import { Microscope } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Button, Panel, Segmented, Switch } from '@/design-system';
import { DataPanel } from './components/DataPanel';
import { DecisionPlane, type EditTool } from './components/DecisionPlane';
import { FeaturePanel } from './components/FeaturePanel';
import { HistogramPanel } from './components/HistogramPanel';
import { LandscapePanel } from './components/LandscapePanel';
import { InspectPanel } from './components/InspectPanel';
import { LossChart } from '@/shared/components/LossChart';
import { LayerStepper, NetworkGraph } from './components/NetworkGraph';
import { StatsStrip } from './components/StatsStrip';
import { PresetBar } from './components/PresetBar';
import { Toolbar } from './components/Toolbar';
import { TrainingPanel } from './components/TrainingPanel';
import { presetConfig } from './presets';
import { usePalette } from '@/shared/canvas/colors';
import { useMlpStore } from './store';
import { CONFIG_PARAM, decodeConfig, encodeConfig } from './url';

/** Keys that should drive the lab only when the user is not typing into / operating a control. */
function isInteractiveTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'A'].includes(target.tagName) ||
    ['slider', 'radio', 'switch', 'tab', 'checkbox'].includes(target.getAttribute('role') ?? '')
  );
}

export default function MlpPage() {
  const palette = usePalette();
  const store = useMlpStore();
  const {
    status,
    error,
    notice,
    config,
    scene,
    snapshot,
    history,
    running,
    speed,
    activePreset,
    inspecting,
    inspection,
    inspectStage,
  } = store;
  const [showTest, setShowTest] = useState(true);
  const [tool, setTool] = useState<EditTool>('look');
  const [activeClass, setActiveClass] = useState(0);
  const [searchParams, setSearchParams] = useSearchParams();
  // Only the URL present when the page opens decides the starting setup.
  const initialParam = useRef(searchParams.get(CONFIG_PARAM));

  useEffect(() => {
    useMlpStore.getState().boot(decodeConfig(initialParam.current) ?? undefined);
    return () => useMlpStore.getState().dispose();
  }, []);

  // Keep the address bar in sync (debounced) so the URL is always a shareable snapshot.
  useEffect(() => {
    if (status !== 'ready') return;
    const timer = window.setTimeout(() => {
      const encoded = encodeConfig(config);
      setSearchParams(encoded ? { [CONFIG_PARAM]: encoded } : {}, { replace: true });
    }, 400);
    return () => window.clearTimeout(timer);
  }, [config, status, setSearchParams]);

  const share = useCallback(async () => {
    const encoded = encodeConfig(useMlpStore.getState().config);
    const url = new URL(window.location.href);
    url.hash = `/mlp${encoded ? `?${CONFIG_PARAM}=${encoded}` : ''}`;
    try {
      await navigator.clipboard.writeText(url.toString());
      return true;
    } catch {
      return false;
    }
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || isInteractiveTarget(event.target))
        return;
      const s = useMlpStore.getState();
      if (event.code === 'Space') {
        event.preventDefault();
        s.toggleRunning();
      } else if (event.key === 'ArrowRight') {
        s.step();
      } else if (event.key.toLowerCase() === 'r') {
        s.reset();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const pickNearest = useCallback((x: number, y: number) => {
    const current = useMlpStore.getState().scene;
    if (!current) return;
    const { xy, target } = current.train;
    let best = 0;
    let bestDistance = Infinity;
    for (let i = 0; i < target.length; i++) {
      const dx = (xy[2 * i] as number) - x;
      const dy = (xy[2 * i + 1] as number) - y;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = i;
      }
    }
    useMlpStore.getState().pickSample(best);
  }, []);

  const last = history[history.length - 1];
  const ready = status === 'ready' && scene && snapshot;

  return (
    <div className="mx-auto max-w-[1600px] px-4 pt-8 pb-20 sm:px-6 lg:px-8">
      <header className="mb-6">
        <p className="eyebrow text-brand">02 · Lab</p>
        <h1 className="mt-2 font-display text-[clamp(1.9rem,3.4vw,2.8rem)] leading-none font-semibold tracking-[-0.03em]">
          MLP <span className="font-serif font-normal italic text-muted">playground</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[0.95rem] text-muted">
          Build a network, pick some data and press <span className="text-fg">Train</span>. Hover
          any neuron to see what it responds to, and any connection to read its weight.
        </p>
      </header>

      {status === 'error' && (
        <div role="alert" className="rounded-2xl border border-danger/40 bg-danger/10 p-5 text-sm">
          <p className="font-medium text-danger">The playground engine stopped.</p>
          <p className="mt-1 text-muted">{error}</p>
          <button
            onClick={() => useMlpStore.getState().boot()}
            className="mt-3 cursor-pointer rounded-full border border-line-strong px-4 py-1.5 text-fg hover:border-brand"
          >
            Restart
          </button>
        </div>
      )}

      {status === 'booting' && (
        <div
          className="space-y-4"
          role="status"
          aria-busy="true"
          aria-label="Loading the playground"
        >
          <div className="h-[74px] animate-pulse rounded-2xl bg-raised/60" />
          <div className="grid gap-4 xl:grid-cols-[300px_1fr_400px]">
            <div className="h-[520px] animate-pulse rounded-2xl bg-raised/60" />
            <div className="h-[520px] animate-pulse rounded-2xl bg-raised/60" />
            <div className="h-[520px] animate-pulse rounded-2xl bg-raised/60" />
          </div>
        </div>
      )}

      {ready && (
        <div className="flex flex-col gap-5">
          <Toolbar
            running={running}
            epoch={last?.epoch ?? 0}
            lr={last?.lr ?? snapshot.stats.lr}
            speed={speed}
            onToggle={store.toggleRunning}
            onStep={store.step}
            onReset={store.reset}
            onSpeed={store.setSpeed}
            onShare={share}
          />

          <PresetBar
            activeId={activePreset}
            onApply={(preset) => store.applyPreset(preset.id, presetConfig(preset), preset.speed)}
          />

          {notice && (
            <p
              role="status"
              className="rounded-xl border border-pos/40 bg-pos/10 px-4 py-2.5 text-sm text-fg"
            >
              {notice}
            </p>
          )}

          {/* On xl the wrappers become three columns; below xl they dissolve (`contents`) and the
              panels reorder so the output comes first. */}
          <div className="flex flex-col gap-5 xl:grid xl:grid-cols-[300px_minmax(0,1fr)_400px] xl:items-start">
            <div className="contents xl:flex xl:flex-col xl:gap-5">
              <Panel title="Data" eyebrow="01" className="order-5 xl:order-none">
                <DataPanel config={config} onPatch={store.patch} />
              </Panel>
              <Panel title="Features" eyebrow="02" className="order-6 xl:order-none">
                <FeaturePanel config={config} onPatch={store.patch} />
              </Panel>
            </div>

            <div className="contents xl:flex xl:flex-col xl:gap-5">
              <Panel
                title="Network"
                eyebrow="03"
                className="order-2 xl:order-none"
                action={
                  <div className="flex flex-wrap items-center gap-3">
                    <Button
                      variant={inspecting ? 'primary' : 'secondary'}
                      size="sm"
                      aria-pressed={inspecting}
                      onClick={() => (inspecting ? store.stopInspect() : store.startInspect())}
                    >
                      <Microscope className="size-4" />
                      {inspecting ? 'Exit inspector' : 'Inspect a step'}
                    </Button>
                    <LayerStepper
                      hidden={config.hidden}
                      onChange={(hidden) => store.patch({ hidden })}
                    />
                  </div>
                }
              >
                <NetworkGraph
                  scene={scene}
                  snapshot={snapshot}
                  config={config}
                  running={running}
                  onPatch={store.patch}
                  inspection={inspecting ? inspection : null}
                  stage={inspectStage}
                />
              </Panel>
              <AnimatePresence>
                {inspecting && inspection && (
                  <div className="order-2 xl:order-none">
                    <InspectPanel
                      inspection={inspection}
                      stage={inspectStage}
                      onStage={store.setStage}
                      onRandomSample={() =>
                        store.startInspect(Math.floor(Math.random() * scene.train.target.length))
                      }
                      onClose={store.stopInspect}
                      scene={scene}
                      snapshot={snapshot}
                      config={config}
                    />
                  </div>
                )}
              </AnimatePresence>
              <Panel title="Training curves" eyebrow="04" className="order-3 xl:order-none">
                <LossChart history={history} problem={scene.problem} />
              </Panel>
            </div>

            <div className="contents xl:flex xl:flex-col xl:gap-5">
              <Panel
                title="Output"
                eyebrow="05"
                className="order-1 xl:order-none"
                action={
                  <Switch label="Test data" checked={showTest} onCheckedChange={setShowTest} />
                }
              >
                <div className="space-y-4">
                  <DecisionPlane
                    scene={scene}
                    snapshot={snapshot}
                    showTest={showTest}
                    selected={inspecting && inspection ? inspection.point : null}
                    onPick={inspecting ? pickNearest : undefined}
                    editTool={inspecting ? 'look' : tool}
                    activeClass={Math.min(activeClass, scene.numClasses - 1)}
                    onEdit={store.editPoints}
                  />
                  {scene.problem === 'classification' && (
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                      <Segmented<EditTool>
                        label="Editing tool"
                        size="sm"
                        fit
                        value={tool}
                        onValueChange={setTool}
                        options={[
                          { value: 'look', label: 'Look' },
                          { value: 'add', label: 'Add' },
                          { value: 'move', label: 'Move' },
                          { value: 'erase', label: 'Erase' },
                        ]}
                      />
                      {tool === 'add' && (
                        <div
                          className="flex items-center gap-1.5"
                          role="group"
                          aria-label="Class to add"
                        >
                          {Array.from({ length: scene.numClasses }, (_, c) => (
                            <button
                              key={c}
                              type="button"
                              aria-label={`Add class ${c}`}
                              aria-pressed={activeClass === c}
                              onClick={() => setActiveClass(c)}
                              className="size-5 cursor-pointer rounded-full border-2 border-surface"
                              style={{
                                background: palette.css.classes[c % palette.css.classes.length],
                                boxShadow: activeClass === c ? '0 0 0 2px var(--brand)' : undefined,
                              }}
                            />
                          ))}
                        </div>
                      )}
                      {scene.edits > 0 && (
                        <Button size="sm" variant="ghost" onClick={store.restorePoints}>
                          Restore the data
                        </Button>
                      )}
                    </div>
                  )}
                  <StatsStrip stats={last ?? snapshot.stats} scene={scene} />
                  <p className="font-mono text-[0.68rem] text-faint">
                    {inspecting
                      ? 'Inspector on: click the plane to choose another training sample'
                      : '● train (filled) · ○ test (ringed) · hover the plane to probe the model'}
                  </p>
                </div>
              </Panel>
              <Panel title="Training" eyebrow="06" className="order-4 xl:order-none">
                <TrainingPanel config={config} onPatch={store.patch} />
              </Panel>
            </div>
          </div>

          <Panel title="Loss landscape" eyebrow="07">
            <LandscapePanel />
          </Panel>
          <Panel title="Weights and gradients" eyebrow="08">
            <HistogramPanel />
          </Panel>
        </div>
      )}
    </div>
  );
}
