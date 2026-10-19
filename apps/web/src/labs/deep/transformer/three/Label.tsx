import { useEffect, useMemo } from 'react';
import * as THREE from 'three';

/** World units per canvas pixel: sets how big the text appears in the scene. */
const WORLD_PER_PX = 0.0034;
const FONT_PX = 40;

interface Props {
  text: string;
  position: [number, number, number];
  /** CSS colour, e.g. `rgb(120, 120, 140)`. */
  color: string;
  /** Rotation in the screen plane, radians (sprites always face the camera). */
  rotation?: number;
  /** 0..1 anchor inside the label: (0, 0) bottom-left, (0.5, 0.5) centre. */
  anchor?: [number, number];
  opacity?: number;
  bold?: boolean;
}

/**
 * Text in the 3-D scene as a camera-facing sprite drawn from a canvas. (drei's `Html` mounts a
 * separate React root per label, which triggers React warnings in development.)
 */
export function Label({
  text,
  position,
  color,
  rotation = 0,
  anchor = [0.5, 0.5],
  opacity = 1,
  bold = false,
}: Props) {
  const { texture, width, height } = useMemo(() => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const font = `${bold ? 600 : 400} ${FONT_PX}px "JetBrains Mono Variable", ui-monospace, monospace`;
    let measured = FONT_PX * text.length * 0.62;
    if (ctx) {
      ctx.font = font;
      measured = ctx.measureText(text).width;
    }
    const pad = 8;
    canvas.width = Math.ceil(measured + pad * 2);
    canvas.height = Math.ceil(FONT_PX * 1.4);
    const draw = canvas.getContext('2d');
    if (draw) {
      draw.font = font;
      draw.textBaseline = 'middle';
      draw.fillStyle = color;
      draw.fillText(text, pad, canvas.height / 2);
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    return { texture: tex, width: canvas.width, height: canvas.height };
  }, [text, color, bold]);

  useEffect(() => () => texture.dispose(), [texture]);

  return (
    <sprite
      position={position}
      scale={[width * WORLD_PER_PX, height * WORLD_PER_PX, 1]}
      center={new THREE.Vector2(anchor[0], anchor[1])}
      renderOrder={10}
    >
      <spriteMaterial
        map={texture}
        transparent
        opacity={opacity}
        rotation={rotation}
        depthTest={false}
        depthWrite={false}
      />
    </sprite>
  );
}
