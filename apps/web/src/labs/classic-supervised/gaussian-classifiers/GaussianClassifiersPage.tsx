import type { Painter } from '@/shared/lab-kit';
import { ClassifierLab, type ClassifierContext } from '@/shared/lab-kit';
import { gaussianOf, gaussianSpec } from './gaussianSpec';

/** Draws each class's 2σ ellipse (and its centre) from the model's means and covariances. */
const ellipses = (ctx: ClassifierContext) => {
  const model = gaussianOf(ctx.run.model);
  if (!model) return undefined;
  return (p: Painter) => {
    model.means.forEach((mu, c) => {
      const cov = model.covariances[c];
      if (!cov) return;
      const a = cov.get(0, 0);
      const b = cov.get(0, 1);
      const d = cov.get(1, 1);
      const tr = a + d;
      const disc = Math.sqrt(Math.max(0, ((a - d) / 2) ** 2 + b * b));
      const l1 = tr / 2 + disc;
      const l2 = Math.max(tr / 2 - disc, 1e-12);
      const angle = Math.abs(b) < 1e-12 && a >= d ? 0 : Math.atan2(l1 - a, b);
      const color = p.palette.css.classes[c % p.palette.css.classes.length] as string;
      const steps = 64;
      p.ctx.beginPath();
      for (let s = 0; s <= steps; s++) {
        const t = (s / steps) * 2 * Math.PI;
        const u = 2 * Math.sqrt(l1) * Math.cos(t);
        const v = 2 * Math.sqrt(l2) * Math.sin(t);
        const x = (mu[0] as number) + u * Math.cos(angle) - v * Math.sin(angle);
        const y = (mu[1] as number) + u * Math.sin(angle) + v * Math.cos(angle);
        if (s === 0) p.ctx.moveTo(p.px(x), p.py(y));
        else p.ctx.lineTo(p.px(x), p.py(y));
      }
      p.ctx.lineWidth = 2;
      p.ctx.strokeStyle = color;
      p.ctx.stroke();
      p.ctx.beginPath();
      p.ctx.arc(p.px(mu[0] as number), p.py(mu[1] as number), 5, 0, 2 * Math.PI);
      p.ctx.fillStyle = color;
      p.ctx.fill();
      p.ctx.strokeStyle = p.palette.css.surface;
      p.ctx.lineWidth = 1.5;
      p.ctx.stroke();
    });
  };
};

export default function GaussianClassifiersPage() {
  return (
    <ClassifierLab
      spec={gaussianSpec}
      text={{
        eyebrow: 'Classic · supervised',
        title: 'Naive Bayes, LDA & QDA',
        accent: 'each class is a Gaussian',
        blurb:
          'Model every class as a bell-shaped cloud and apply Bayes’ rule. The only difference between the three models is which covariance they allow, and the ellipses show it.',
      }}
      overlay={ellipses}
      howItWorks={
        <div className="space-y-3 text-sm text-muted">
          <p>
            Each class c gets a Gaussian N(μ_c, Σ_c) and a prior π_c. A point is assigned to the
            class with the largest π_c · N(x; μ_c, Σ_c). The ellipses are the 2σ contours of those
            Gaussians; the dots are their centres.
          </p>
          <p>
            Naive Bayes keeps only the variances (axis-aligned ellipses): few parameters, robust,
            but blind to correlation. LDA shares one covariance, so the quadratic terms cancel and
            the boundary is a straight line. QDA gives each class its own, so the boundary is a
            conic, at the cost of many more parameters; shrinkage steadies them.
          </p>
        </div>
      }
    />
  );
}
