import { describe, expect, it } from 'vitest';
import { activeTracks, labsOf, modules, tracks } from './registry';

describe('lab registry', () => {
  it('has unique ids and unique routes', () => {
    const ids = modules.map((m) => m.id);
    expect(new Set(ids).size).toBe(ids.length);
    const paths = modules.flatMap((m) => (m.path ? [m.path] : []));
    expect(new Set(paths).size).toBe(paths.length);
  });

  it('describes every lab completely', () => {
    for (const m of modules) {
      expect(m.title.length).toBeGreaterThan(0);
      expect(m.tagline.length).toBeGreaterThan(10);
      expect(m.topics.length).toBeGreaterThan(0);
      if (m.status === 'live') expect(m.path).toMatch(/^\//);
    }
  });

  it('puts every lab in a known track, with the labs of a track together', () => {
    const known = new Set(tracks.map((t) => t.id));
    expect(modules.every((m) => known.has(m.track))).toBe(true);
    // labs of one track are contiguous in the list, in the order of the tracks
    const order = modules.map((m) => tracks.findIndex((t) => t.id === m.track));
    expect([...order].sort((a, b) => a - b)).toEqual(order);
  });

  it('points every live lab at a page file that exists', () => {
    const files = Object.keys(import.meta.glob('../labs/**/*Page.tsx')).map((f) =>
      f.replace('../labs/', '').replace(/\.tsx$/, ''),
    );
    for (const m of modules) if (m.status === 'live') expect(files).toContain(m.page);
  });

  it('knows which tracks have labs', () => {
    expect(activeTracks().length).toBeGreaterThan(0);
    for (const t of activeTracks()) expect(labsOf(t.id).length).toBeGreaterThan(0);
    expect(new Set(tracks.map((t) => t.id)).size).toBe(tracks.length);
  });

  it('starts with the beginner labs (pedagogical order)', () => {
    expect(modules[0]?.level).toBe('Beginner');
    expect(labsOf('foundations')[0]?.level).toBe('Beginner');
  });
});
