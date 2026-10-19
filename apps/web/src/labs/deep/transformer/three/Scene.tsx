import { OrbitControls } from '@react-three/drei';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import type { ForwardTrace, NextTokenDistribution } from '@sim/deep';
import { useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { mix, usePalette, type Palette } from '@/shared/canvas/colors';
import { divergingRgb } from '@/shared/components/Heat';
import { tokenLabel } from '../tokens';
import { Label } from './Label';
import {
  COLUMN_W,
  HEAD_SIZE,
  SLAB_GAP,
  SLAB_H,
  cellIndex,
  cellOf,
  cellPosition,
  columnX,
  headCells,
  headOrigin,
  outputBars,
  outputZ,
  slabScale,
  type HeadCell,
  type Slab,
  type Stage,
} from './layout';

export type HoverInfo =
  | { kind: 'cell'; slab: Slab; token: number; row: number; value: number; x: number; y: number }
  | { kind: 'head'; layer: number; head: number; cell: HeadCell; x: number; y: number }
  | { kind: 'bar'; id: number; prob: number; kept: boolean; x: number; y: number };

interface Props {
  trace: ForwardTrace;
  slabs: Slab[];
  stages: Stage[];
  /** Active stage index, or null for the overview. */
  stage: number | null;
  /** Token position in focus. */
  selected: number;
  tokenStrings: string[];
  distribution: NextTokenDistribution;
  tokenName: (id: number) => string;
  reducedMotion: boolean;
  onSelect: (token: number) => void;
  onHover: (info: HoverInfo | null) => void;
}

const DEPTH = 0.05;
const box = new THREE.BoxGeometry(1, 1, 1);
const columnEdges = new THREE.EdgesGeometry(
  new THREE.BoxGeometry(COLUMN_W * 1.05, SLAB_H * 1.02, 0.1),
);

const color = (c: readonly [number, number, number]) =>
  new THREE.Color().setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace);

/**
 * Dimmed stages fade to this opacity. Materials are ALWAYS `transparent`: three.js compiles an
 * opaque shader variant for non-transparent materials and does not recompile when the flag flips.
 */
const dimOpacity = 0.14;

// ───────────────────────────────── slabs ─────────────────────────────────

function SlabMesh({
  slab,
  palette,
  opacity,
  onHover,
  onSelect,
}: {
  slab: Slab;
  palette: Palette;
  opacity: number;
  onHover: Props['onHover'];
  onSelect: Props['onSelect'];
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const count = slab.tokens * slab.rows;

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    const scale = slabScale(slab.values);
    const position = new THREE.Vector3();
    const size = new THREE.Vector3();
    const identity = new THREE.Quaternion();
    for (let t = 0; t < slab.tokens; t++) {
      for (let r = 0; r < slab.rows; r++) {
        const i = cellIndex(t, r, slab.rows);
        const p = cellPosition(slab, t, r);
        m.compose(position.set(p.x, p.y, p.z), identity, size.set(p.w, p.h, DEPTH));
        mesh.setMatrixAt(i, m);
        mesh.setColorAt(i, color(divergingRgb(slab.values[i] as number, scale, palette)));
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [slab, palette]);

  const hit = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    if (e.instanceId === undefined) return;
    const { token, row } = cellOf(e.instanceId, slab.rows);
    onHover({
      kind: 'cell',
      slab,
      token,
      row,
      value: slab.values[e.instanceId] as number,
      x: e.nativeEvent.offsetX,
      y: e.nativeEvent.offsetY,
    });
  };

  return (
    <instancedMesh
      key={count}
      ref={ref}
      args={[box, undefined, count]}
      frustumCulled={false}
      onPointerMove={hit}
      onPointerOut={() => onHover(null)}
      onClick={(e) => {
        e.stopPropagation();
        if (e.instanceId !== undefined) onSelect(cellOf(e.instanceId, slab.rows).token);
      }}
    >
      <meshLambertMaterial transparent opacity={opacity} depthWrite={opacity >= 0.99} />
    </instancedMesh>
  );
}

// ───────────────────────────── attention heads ─────────────────────────────

function HeadMesh({
  cells,
  layer,
  head,
  z,
  selected,
  palette,
  opacity,
  onHover,
}: {
  cells: HeadCell[];
  layer: number;
  head: number;
  z: number;
  selected: number;
  palette: Palette;
  opacity: number;
  onHover: Props['onHover'];
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const side = cells.length > 0 ? HEAD_SIZE / Math.max(...cells.map((c) => c.query + 1)) : 0.1;

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const size = new THREE.Vector3();
    const identity = new THREE.Quaternion();
    cells.forEach((c, i) => {
      const inFocus = c.query === selected;
      const s = side * (inFocus ? 0.98 : 0.8);
      m.compose(position.set(c.x, c.y, z), identity, size.set(s, s, DEPTH * (inFocus ? 2.2 : 1)));
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(
        i,
        color(mix(palette.surface, palette.brand, Math.sqrt(Math.min(1, c.weight)))),
      );
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [cells, selected, palette, z, side]);

  return (
    <instancedMesh
      key={cells.length}
      ref={ref}
      args={[box, undefined, cells.length]}
      frustumCulled={false}
      onPointerMove={(e) => {
        e.stopPropagation();
        const cell = e.instanceId === undefined ? undefined : cells[e.instanceId];
        if (cell) {
          onHover({
            kind: 'head',
            layer,
            head,
            cell,
            x: e.nativeEvent.offsetX,
            y: e.nativeEvent.offsetY,
          });
        }
      }}
      onPointerOut={() => onHover(null)}
    >
      <meshLambertMaterial transparent opacity={opacity} depthWrite={opacity >= 0.99} />
    </instancedMesh>
  );
}

// ───────────────────────────────── camera ─────────────────────────────────

interface Pose {
  position: THREE.Vector3;
  target: THREE.Vector3;
}

function poseFor(stage: Stage | null, depth: number): Pose {
  if (!stage) {
    return {
      position: new THREE.Vector3(5.6, 2.7, 6.2),
      target: new THREE.Vector3(0, -0.5, depth * 0.42),
    };
  }
  // Distances are chosen so the whole object of the stage fits the frame (vertical fov 36°).
  if (stage.output) {
    // bars stand on the floor of the tunnel and their labels hang underneath
    return {
      position: new THREE.Vector3(3.6, 0.2, stage.z + 7.6),
      target: new THREE.Vector3(0, -1.0, stage.z),
    };
  }
  if (stage.heads !== undefined) {
    // the slab plus the head matrices hanging beneath it
    return {
      position: new THREE.Vector3(3.8, -0.4, stage.z + 9.6),
      target: new THREE.Vector3(0, -1.0, stage.z),
    };
  }
  const wide = stage.slabIds.length > 1; // e.g. MLP hidden + MLP output
  return {
    position: new THREE.Vector3(3.6, 0.9, stage.z + (wide ? 8.4 : 7.2)),
    target: new THREE.Vector3(0, 0, stage.z - (wide ? SLAB_GAP / 2 : 0.2)),
  };
}

/** Orbit controls plus an eased fly-to whenever the stage changes (user input cancels it). */
function CameraRig({ pose: basePose, animate }: { pose: Pose; animate: boolean }) {
  const controls = useRef<OrbitControlsImpl>(null);
  const flying = useRef(true);
  const aspect = useThree((state) => state.size.width / Math.max(1, state.size.height));

  // Tall, narrow screens see less sideways: pull the camera back so the stage still fits.
  const pose = useMemo<Pose>(() => {
    const back = Math.max(1, 1.35 / aspect);
    return {
      position: basePose.target
        .clone()
        .add(basePose.position.clone().sub(basePose.target).multiplyScalar(back)),
      target: basePose.target,
    };
  }, [basePose, aspect]);

  useEffect(() => {
    flying.current = true;
  }, [pose]);

  useFrame(({ camera }, delta) => {
    const c = controls.current;
    if (!c || !flying.current) return;
    const k = animate ? 1 - Math.exp(-delta * 3.4) : 1;
    camera.position.lerp(pose.position, k);
    c.target.lerp(pose.target, k);
    c.update();
    if (
      camera.position.distanceTo(pose.position) < 0.01 &&
      c.target.distanceTo(pose.target) < 0.01
    ) {
      flying.current = false;
    }
  });

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.09}
      minDistance={2}
      maxDistance={34}
      onStart={() => {
        flying.current = false;
      }}
    />
  );
}

// ───────────────────────────────── scene ─────────────────────────────────

export default function Scene({
  trace,
  slabs,
  stages,
  stage,
  selected,
  tokenStrings,
  distribution,
  tokenName,
  reducedMotion,
  onSelect,
  onHover,
}: Props) {
  const palette = usePalette();
  const active = stage === null ? null : stages[stage];
  const activeSlabs = useMemo(() => new Set(active?.slabIds ?? []), [active]);
  const depth = (slabs[slabs.length - 1]?.z ?? 0) - 1.5;
  const pose = useMemo(() => poseFor(active ?? null, depth), [active, depth]);
  const T = trace.ids.length;
  const bars = useMemo(() => outputBars(distribution), [distribution]);
  const barZ = outputZ(slabs);

  const heads = useMemo(
    () =>
      trace.layers.map((layer, l) => ({
        layer: l,
        z: slabs.find((s) => s.id === `attn-${l}`)?.z ?? 0,
        heads: layer.probs.map((probs, h) => ({
          head: h,
          cells: headCells(probs, h, layer.probs.length),
          origin: headOrigin(h, layer.probs.length),
        })),
      })),
    [trace, slabs],
  );

  const slabOpacity = (id: string) => (stage === null || activeSlabs.has(id) ? 1 : dimOpacity);
  const headOpacity = (layer: number) =>
    stage === null || active?.heads === layer ? 1 : dimOpacity;
  const outputOpacity = stage === null || active?.output ? 1 : dimOpacity;
  const columnLines = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: color(palette.brand),
        transparent: true,
        opacity: 0.95,
      }),
    [palette],
  );

  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ fov: 36, position: [6.4, 3.1, 8.2], near: 0.1, far: 120 }}
      gl={{ alpha: true, antialias: true }}
      onPointerMissed={() => onHover(null)}
    >
      <ambientLight intensity={1.5} />
      <directionalLight position={[3, 6, 8]} intensity={0.9} />
      <CameraRig pose={pose} animate={!reducedMotion} />

      {slabs.map((slab) => (
        <group key={slab.id}>
          <SlabMesh
            slab={slab}
            palette={palette}
            opacity={slabOpacity(slab.id)}
            onHover={onHover}
            onSelect={onSelect}
          />
          {selected >= 0 && selected < slab.tokens && slabOpacity(slab.id) === 1 && (
            <lineSegments
              geometry={columnEdges}
              material={columnLines}
              position={[columnX(selected, slab.tokens), 0, slab.z]}
            />
          )}
        </group>
      ))}

      {/* token labels above the embedding slab */}
      {tokenStrings.map((label, t) => (
        <Label
          key={t}
          text={tokenLabel(label, trace.ids[t] as number)}
          position={[columnX(t, T), SLAB_H / 2 + 0.08, 0]}
          color={t === selected ? palette.css.fg : palette.css.muted}
          bold={t === selected}
          anchor={[0, 0.5]}
          rotation={Math.PI / 3.2}
        />
      ))}

      {/* attention heads under each attention slab */}
      {heads.map(({ layer, z, heads: hs }) =>
        hs.map(({ head, cells, origin }) => (
          <group key={`${layer}-${head}`}>
            <HeadMesh
              cells={cells}
              layer={layer}
              head={head}
              z={z}
              selected={selected}
              palette={palette}
              opacity={headOpacity(layer)}
              onHover={onHover}
            />
            <Label
              text={`layer ${layer + 1} · head ${head + 1}`}
              position={[origin.x, origin.y - HEAD_SIZE / 2 - 0.14, z]}
              color={palette.css.muted}
              opacity={headOpacity(layer) === 1 ? 1 : 0.35}
            />
          </group>
        )),
      )}

      {/* next-token probabilities */}
      {bars.map((b) => (
        <group key={b.id} position={[b.x, 0, barZ]}>
          <mesh
            position={[0, -SLAB_H / 2 + b.height / 2, 0]}
            onPointerMove={(e) => {
              e.stopPropagation();
              onHover({
                kind: 'bar',
                id: b.id,
                prob: b.prob,
                kept: b.kept,
                x: e.nativeEvent.offsetX,
                y: e.nativeEvent.offsetY,
              });
            }}
            onPointerOut={() => onHover(null)}
          >
            <boxGeometry args={[0.24, b.height, 0.24]} />
            <meshLambertMaterial
              color={color(b.kept ? palette.brand : mix(palette.surface, palette.fg, 0.35))}
              transparent
              opacity={outputOpacity}
            />
          </mesh>
          <Label
            text={tokenName(b.id)}
            position={[0, -SLAB_H / 2 - 0.08, 0]}
            color={palette.css.muted}
            anchor={[1, 0.5]}
            rotation={Math.PI / 4}
            opacity={outputOpacity === 1 ? 1 : 0.35}
          />
        </group>
      ))}
    </Canvas>
  );
}
