import { useEffect, useState } from 'react';
import 'katex/dist/katex.min.css';

/**
 * Renders a KaTeX formula. KaTeX itself is loaded on demand (separate chunk), so pages that
 * never show math don't pay for it. Only formulas defined in code are rendered here.
 */
export function Tex({ tex, block = false }: { tex: string; block?: boolean }) {
  const [html, setHtml] = useState('');

  useEffect(() => {
    let alive = true;
    void import('katex').then(({ default: katex }) => {
      if (!alive) return;
      setHtml(
        katex.renderToString(tex, { throwOnError: false, displayMode: block, output: 'html' }),
      );
    });
    return () => {
      alive = false;
    };
  }, [tex, block]);

  // Reserve the formula's source as the accessible name until (and after) it is typeset.
  return <span role="math" aria-label={tex} dangerouslySetInnerHTML={{ __html: html || tex }} />;
}
