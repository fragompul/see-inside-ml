import {
  LR_SCHEDULES,
  OPTIMIZER_NAMES,
  REGRESSION_LOSSES,
  lossInfo,
  optimizerInfo,
  scheduleInfo,
} from '@sim/core';
import { datasetInfo, type PlaygroundConfig, type RegularizationKind } from '@sim/deep';
import { Segmented, Select, Slider, Switch, Tex } from '@/design-system';
import { BATCH_LADDER, LR_LADDER, REG_LADDER, nearestIndex } from '@/shared/lab/ladders';

const trim = (v: number) => String(Number(v.toPrecision(3)));

export function TrainingPanel({
  config,
  onPatch,
}: {
  config: PlaygroundConfig;
  onPatch: (patch: Partial<PlaygroundConfig>) => void;
}) {
  const regression = datasetInfo[config.dataset].problem === 'regression';
  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-2">
        <span className="text-[0.82rem] text-muted">Optimizer</span>
        <Segmented
          label="Optimizer"
          size="sm"
          value={config.optimizer}
          onValueChange={(optimizer) => onPatch({ optimizer })}
          options={OPTIMIZER_NAMES.map((o) => ({ value: o, label: optimizerInfo[o].label }))}
        />
        <p className="text-[0.72rem] leading-snug text-faint">
          {optimizerInfo[config.optimizer].note}
        </p>
      </div>

      <Slider
        label="Learning rate"
        min={0}
        max={LR_LADDER.length - 1}
        step={1}
        value={nearestIndex(LR_LADDER, config.lr)}
        onValueChange={(i) => onPatch({ lr: LR_LADDER[i] as number })}
        format={() => trim(config.lr)}
      />

      <Select
        label="Learning-rate schedule"
        value={config.schedule}
        onValueChange={(schedule) => onPatch({ schedule })}
        options={LR_SCHEDULES.map((s) => ({ value: s, label: scheduleInfo[s].label }))}
      />

      <Slider
        label="Batch size"
        min={0}
        max={BATCH_LADDER.length - 1}
        step={1}
        value={Math.max(
          0,
          BATCH_LADDER.findIndex((b) => b >= config.batchSize) === -1
            ? BATCH_LADDER.length - 1
            : BATCH_LADDER.findIndex((b) => b >= config.batchSize),
        )}
        onValueChange={(i) => onPatch({ batchSize: BATCH_LADDER[i] as number })}
        format={() => String(config.batchSize)}
      />

      <div className="flex flex-col gap-2">
        <span className="text-[0.82rem] text-muted">Regularization</span>
        <Segmented<RegularizationKind>
          label="Regularization"
          size="sm"
          value={config.regularization}
          onValueChange={(regularization) => onPatch({ regularization })}
          options={[
            { value: 'none', label: 'None' },
            { value: 'l1', label: 'L1' },
            { value: 'l2', label: 'L2' },
          ]}
        />
        {config.regularization !== 'none' && (
          <Slider
            label="Rate λ"
            min={0}
            max={REG_LADDER.length - 1}
            step={1}
            value={nearestIndex(REG_LADDER, config.regularizationRate)}
            onValueChange={(i) => onPatch({ regularizationRate: REG_LADDER[i] as number })}
            format={() => trim(config.regularizationRate)}
          />
        )}
      </div>

      {regression && (
        <Select
          label="Loss function"
          value={config.loss}
          onValueChange={(loss) => onPatch({ loss })}
          options={REGRESSION_LOSSES.map((l) => ({ value: l, label: lossInfo[l].label }))}
        />
      )}
      {regression && (
        <p className="-mt-2 text-[0.8rem] text-muted">
          <Tex tex={`L = ${lossInfo[config.loss].formula}`} />
        </p>
      )}

      <Switch
        label="Skip connections (inputs feed every layer)"
        checked={config.skipConnections}
        onCheckedChange={(skipConnections) => onPatch({ skipConnections })}
      />
    </div>
  );
}
