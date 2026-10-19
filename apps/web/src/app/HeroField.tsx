import { useEffect, useRef } from 'react';
import { useTheme } from '@/design-system';
import { buildField } from './field';

const LAYERS = [3, 6, 7, 6, 3];
const FORWARD_SECONDS = 6.5;
const BACKWARD_SECONDS = 3.5;
const MAX_PULSES = 70;

interface Pulse {
  edge: number;
  /** 0 = source node, 1 = target node. */
  t: number;
  dir: 1 | -1;
  speed: number;
}

/**
 * The hero's living network. It is not decoration: it replays what the product teaches.
 * Signals travel layer to layer (forward pass, coloured by weight sign), then a violet wave
 * flows back (backpropagation). The pointer attracts nodes and lights up nearby connections.
 */
export function HeroField({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { resolved } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const field = buildField(LAYERS);
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const styles = getComputedStyle(document.documentElement);
    const read = (name: string, fallback: string) =>
      styles.getPropertyValue(name).trim() || fallback;
    const colors = {
      pos: read('--pos', '#ffb454'),
      neg: read('--neg', '#4fd1ff'),
      brand: read('--brand', '#9a8cff'),
      bg: read('--bg', '#08090e'),
    };

    let width = 0;
    let height = 0;
    const pointer = { x: 0, y: 0, active: false };
    const glow = new Float32Array(field.nodes.length);
    const tint: string[] = field.nodes.map((n) => (n.phase < Math.PI ? colors.pos : colors.neg));
    const base: string[] = [...tint];
    const pulses: Pulse[] = [];
    const px = new Float32Array(field.nodes.length);
    const py = new Float32Array(field.nodes.length);

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width;
      height = rect.height;
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const layout = (time: number) => {
      const wide = width >= 900;
      const x0 = wide ? width * 0.44 : width * 0.06;
      const x1 = wide ? width * 0.95 : width * 0.94;
      const y0 = height * (wide ? 0.16 : 0.1);
      const y1 = height * (wide ? 0.84 : 0.9);
      field.nodes.forEach((node, i) => {
        let x = x0 + node.x * (x1 - x0);
        let y = y0 + node.y * (y1 - y0);
        x += Math.cos(time * 0.0006 + node.phase) * 3;
        y += Math.sin(time * 0.0008 + node.phase * 1.3) * 3.5;
        if (pointer.active) {
          const dx = pointer.x - x;
          const dy = pointer.y - y;
          const dist = Math.hypot(dx, dy);
          const radius = 150;
          if (dist < radius && dist > 0.001) {
            const pull = (1 - dist / radius) ** 2 * 12;
            x += (dx / dist) * pull;
            y += (dy / dist) * pull;
          }
        }
        px[i] = x;
        py[i] = y;
      });
    };

    const spawn = (edge: number, dir: 1 | -1) => {
      if (pulses.length >= MAX_PULSES) return;
      pulses.push({
        edge,
        t: dir === 1 ? 0 : 1,
        dir,
        speed: 0.55 + Math.random() * 0.4,
      });
    };

    const pick = (list: number[] | undefined) =>
      list && list.length > 0 ? list[Math.floor(Math.random() * list.length)] : undefined;

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Connections
      field.edges.forEach((edge) => {
        const ax = px[edge.from] ?? 0;
        const ay = py[edge.from] ?? 0;
        const bx = px[edge.to] ?? 0;
        const by = py[edge.to] ?? 0;
        const strength = Math.abs(edge.weight);
        let alpha = 0.07 + 0.2 * strength;
        if (pointer.active) {
          const d = Math.hypot(pointer.x - (ax + bx) / 2, pointer.y - (ay + by) / 2);
          alpha += Math.max(0, 1 - d / 200) * 0.4;
        }
        ctx.globalAlpha = Math.min(alpha, 0.85);
        ctx.strokeStyle = edge.weight > 0 ? colors.pos : colors.neg;
        ctx.lineWidth = 0.6 + 1.1 * strength;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      });

      // Signals in flight
      pulses.forEach((pulse) => {
        const edge = field.edges[pulse.edge];
        if (!edge) return;
        const ax = px[edge.from] ?? 0;
        const ay = py[edge.from] ?? 0;
        const bx = px[edge.to] ?? 0;
        const by = py[edge.to] ?? 0;
        const x = ax + (bx - ax) * pulse.t;
        const y = ay + (by - ay) * pulse.t;
        const tail = pulse.t - pulse.dir * 0.14;
        const color = pulse.dir === 1 ? (edge.weight > 0 ? colors.pos : colors.neg) : colors.brand;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.55;
        ctx.beginPath();
        ctx.moveTo(ax + (bx - ax) * tail, ay + (by - ay) * tail);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.14;
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 0.95;
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      });

      // Neurons
      field.nodes.forEach((_, i) => {
        const g = glow[i] ?? 0;
        const x = px[i] ?? 0;
        const y = py[i] ?? 0;
        const color = tint[i] ?? colors.brand;
        const r = 4.4 + g * 3;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.07 + 0.2 * g;
        ctx.beginPath();
        ctx.arc(x, y, r * 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = colors.bg;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.4;
        ctx.globalAlpha = 0.45 + 0.55 * g;
        ctx.stroke();
        if (g > 0.02) {
          ctx.fillStyle = color;
          ctx.globalAlpha = g;
          ctx.beginPath();
          ctx.arc(x, y, r * 0.55, 0, Math.PI * 2);
          ctx.fill();
        }
      });
      ctx.globalAlpha = 1;
    };

    resize();

    // Reduced motion: a single, still frame with a few "active" neurons.
    if (reduceMotion) {
      [2, 5, 9, 13, 17, 22].forEach((i) => {
        if (i < glow.length) glow[i] = 0.9;
      });
      layout(0);
      draw();
      const observer = new ResizeObserver(() => {
        resize();
        layout(0);
        draw();
      });
      observer.observe(canvas);
      return () => observer.disconnect();
    }

    let raf = 0;
    let last = performance.now();
    let spawnClock = 0;
    let modeClock = 0;
    let mode: 'forward' | 'backward' = 'forward';
    let visible = true;

    const frame = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;

      modeClock += dt;
      const limit = mode === 'forward' ? FORWARD_SECONDS : BACKWARD_SECONDS;
      if (modeClock > limit) {
        modeClock = 0;
        mode = mode === 'forward' ? 'backward' : 'forward';
      }

      spawnClock += dt;
      const interval = mode === 'forward' ? 0.22 : 0.1;
      while (spawnClock > interval) {
        spawnClock -= interval;
        if (mode === 'forward') {
          const source = pick(field.layers[0]);
          const edge = source === undefined ? undefined : pick(field.outgoing[source]);
          if (edge !== undefined) spawn(edge, 1);
        } else {
          const source = pick(field.layers[field.layers.length - 1]);
          const edge = source === undefined ? undefined : pick(field.incoming[source]);
          if (edge !== undefined) spawn(edge, -1);
        }
      }

      for (let i = pulses.length - 1; i >= 0; i--) {
        const pulse = pulses[i];
        if (!pulse) continue;
        pulse.t += pulse.dir * pulse.speed * dt;
        const arrived = pulse.dir === 1 ? pulse.t >= 1 : pulse.t <= 0;
        if (!arrived) continue;
        const edge = field.edges[pulse.edge];
        pulses.splice(i, 1);
        if (!edge) continue;
        const node = pulse.dir === 1 ? edge.to : edge.from;
        glow[node] = 1;
        tint[node] = pulse.dir === 1 ? (edge.weight > 0 ? colors.pos : colors.neg) : colors.brand;
        const next = pick(pulse.dir === 1 ? field.outgoing[node] : field.incoming[node]);
        if (next !== undefined && Math.random() < 0.9) spawn(next, pulse.dir);
      }

      const decay = Math.exp(-dt * 2.2);
      for (let i = 0; i < glow.length; i++) {
        const value = (glow[i] ?? 0) * decay;
        glow[i] = value;
        if (value < 0.03) tint[i] = base[i] ?? colors.brand;
      }

      layout(now);
      draw();
      if (visible) raf = requestAnimationFrame(frame);
    };

    const start = () => {
      cancelAnimationFrame(raf);
      last = performance.now();
      raf = requestAnimationFrame(frame);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);

    const intersection = new IntersectionObserver(([entry]) => {
      visible = entry?.isIntersecting ?? true;
      if (visible) start();
      else cancelAnimationFrame(raf);
    });
    intersection.observe(canvas);

    const onMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      pointer.x = event.clientX - rect.left;
      pointer.y = event.clientY - rect.top;
      pointer.active =
        pointer.x >= 0 && pointer.y >= 0 && pointer.x <= rect.width && pointer.y <= rect.height;
    };
    const onLeave = () => {
      pointer.active = false;
    };
    window.addEventListener('pointermove', onMove, { passive: true });
    document.documentElement.addEventListener('pointerleave', onLeave);

    start();

    return () => {
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      intersection.disconnect();
      window.removeEventListener('pointermove', onMove);
      document.documentElement.removeEventListener('pointerleave', onLeave);
    };
  }, [resolved]);

  return <canvas ref={canvasRef} aria-hidden className={className} />;
}
