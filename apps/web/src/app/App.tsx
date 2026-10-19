import { lazy, Suspense, type ComponentType } from 'react';
import { HashRouter, Route, Routes } from 'react-router-dom';
import { TooltipProvider, useThemeSync } from '@/design-system';
import { Home } from './Home';
import { Layout } from './Layout';
import { modules } from './registry';

// Every lab is a page component under `src/labs/` named in the registry; Vite turns each one into
// its own chunk, fetched only when its route is opened.
const loaders = import.meta.glob<{ default: ComponentType }>('../labs/**/*Page.tsx');
const pages = new Map(
  Object.entries(loaders).map(([file, load]) => [
    file.replace('../labs/', '').replace(/\.tsx$/, ''),
    lazy(load),
  ]),
);

const DesignSystem = lazy(() =>
  import('./DesignSystem').then((m) => ({ default: m.DesignSystem })),
);

export function App() {
  useThemeSync();
  return (
    <TooltipProvider>
      {/* HashRouter keeps deep links working on static hosts such as GitHub Pages. */}
      <HashRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Home />} />
            <Route
              path="design"
              element={
                <Suspense fallback={null}>
                  <DesignSystem />
                </Suspense>
              }
            />
            {modules.map((m) => {
              const Page = m.page ? pages.get(m.page) : undefined;
              if (!m.path || !Page) return null;
              return (
                <Route
                  key={m.id}
                  path={m.path.replace(/^\//, '')}
                  element={
                    <Suspense fallback={null}>
                      <Page />
                    </Suspense>
                  }
                />
              );
            })}
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </HashRouter>
    </TooltipProvider>
  );
}

function NotFound() {
  return (
    <div className="mx-auto max-w-3xl px-5 py-32 text-center">
      <p className="eyebrow">404</p>
      <h1 className="mt-4 font-display text-4xl font-semibold tracking-tight">
        Lost in latent space
      </h1>
      <p className="mt-3 text-muted">That page doesn&apos;t exist. Head back home.</p>
    </div>
  );
}
