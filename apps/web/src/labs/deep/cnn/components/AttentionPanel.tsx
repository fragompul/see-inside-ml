import { IMAGE_SIZE, argmax, evidenceMap, gradCam } from '@sim/deep';
import { useMemo, useState } from 'react';
import { Select } from '@/design-system';
import { PixelCanvas } from '@/shared/components/PixelCanvas';
import { cnnSession, effectiveImage, useCnnStore } from '../store';

/** Two answers to "where was the network looking?": Grad-CAM and its exact, position-wise version. */
export function AttentionPanel() {
  const image = useCnnStore((s) => s.image);
  const center = useCnnStore((s) => s.center);
  const source = useCnnStore((s) => s.source);
  const version = useCnnStore((s) => s.version);
  const trace = useCnnStore((s) => s.trace);
  const [which, setWhich] = useState<string>('auto');
  const input = useMemo(() => effectiveImage(image, center), [image, center]);
  const predicted = trace ? argmax(trace.probs) : 0;
  const target = which === 'auto' ? predicted : Number(which);

  const maps = useMemo(() => {
    if (source === 'blank') return null;
    const net = cnnSession().network;
    return { evidence: evidenceMap(net, input, target), cam: gradCam(net, input, target) };
    // `version` bumps when the weights change while training
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, target, source, version]);

  if (!maps) {
    return (
      <p className="text-sm text-muted">
        Draw a digit or load a test digit above to see where the network looks.
      </p>
    );
  }
  const camEmpty = Math.max(...maps.cam.coarse) === 0;
  return (
    <div className="space-y-6">
      <div className="max-w-xs">
        <Select
          label="Evidence for"
          value={which}
          onValueChange={setWhich}
          options={[
            { value: 'auto', label: `The network’s answer (${predicted})` },
            ...Array.from({ length: 10 }, (_, d) => ({ value: String(d), label: `Digit ${d}` })),
          ]}
        />
      </div>
      <div className="flex flex-wrap gap-8">
        <figure className="space-y-2">
          <PixelCanvas
            values={input}
            side={IMAGE_SIZE}
            size={200}
            overlay={maps.evidence.map}
            ariaLabel="Image with the evidence map: pixels that argue for the chosen digit are washed in ember"
          />
          <figcaption className="max-w-[200px] text-[0.74rem] text-faint">
            <strong className="text-muted">Evidence map</strong> (gradient × activation): exactly
            where the score for {target} comes from.
          </figcaption>
        </figure>
        <figure className="space-y-2">
          <PixelCanvas
            values={input}
            side={IMAGE_SIZE}
            size={200}
            overlay={camEmpty ? null : maps.cam.map}
            ariaLabel="Image with the Grad-CAM map"
          />
          <figcaption className="max-w-[200px] text-[0.74rem] text-faint">
            <strong className="text-muted">Grad-CAM</strong>:{' '}
            {camEmpty
              ? 'empty here — averaging the gradients cancelled the evidence.'
              : 'channel-averaged gradients, the classic recipe.'}
          </figcaption>
        </figure>
      </div>
      <p className="max-w-2xl text-sm text-muted" data-testid="attention-note">
        The last layer of this network is a plain dense layer, so the score for a digit is a sum
        over positions and feature maps of weight × activation. The evidence map shows each
        position’s share of that sum (it adds up to the score exactly). Grad-CAM first averages the
        weights over position, which works when gradients are similar everywhere (networks with
        global pooling) but can blur or even erase the picture here, because this dense layer cares
        where things are. Pixels shown are the {evidenceNote(maps.evidence.score)}
      </p>
    </div>
  );
}

const evidenceNote = (score: number) =>
  `positive part of the evidence; the total evidence for this digit is ${score.toFixed(2)} (before the bias).`;
