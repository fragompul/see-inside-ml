# Neural Network Playground — Plan de trabajo

> Documento de planificación (en español). **La app y su contenido se escriben en inglés.**
> Estado: BORRADOR para discutir. Nada de esto está implementado todavía. Sin commits hasta que se entregue `COMMIT_PLAN.md`.

## 1. Visión

Un laboratorio interactivo, visualmente espectacular, donde cualquiera puede **ver por dentro** cómo funcionan, entrenan e infieren los modelos de IA: desde un perceptrón hasta un transformer, GANs o difusión.

Doble objetivo, con el mismo peso:

| Formativo                                                                   | Portfolio                                              |
| --------------------------------------------------------------------------- | ------------------------------------------------------ |
| Todo es manipulable (arrastrar, editar, hover, pausar, avanzar paso a paso) | Estética de producto real, no de demo                  |
| Cada visualización explica _qué_, _por qué_ y _la matemática_               | Código limpio, tipado, testeado, CI verde              |
| Curva de aprendizaje guiada (beginner → advanced)                           | Deploy público con un clic (demo en vivo en el README) |

**Criterio de éxito #1: si no es bonito, no hemos conseguido nada.** La calidad visual es un requisito de aceptación de cada fase, no un extra final.

## 2. Estado actual (punto de partida)

- `neural_network.py`: MLP vectorizado en NumPy (forward/backward, SGD/Momentum/RMSprop/Adam, L1/L2, skip connections, schedulers).
- `dataset.py`: datasets sintéticos 2D (clasificación y regresión) + feature engineering.
- `visuals.py` + `app.py`: Streamlit + Plotly + SVG inyectado. Limitaciones: rerun del script, sin animación fina, sin estado de cliente, difícil de escalar a muchos modelos.

Se conserva como **referencia matemática** (ver decisión D2), no como UI.

## 3. Decisiones de diseño (a cerrar juntos antes de arrancar)

### D1 — ¿Dónde se ejecuta el entrenamiento/inferencia?

| Opción                                                                              | Pros                                                                                                                                                                       | Contras                                                                                                      |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **A. 100 % en el navegador** (TypeScript + Web Workers, WebGPU/WASM para lo pesado) | Cero latencia, animaciones a 60 fps ligadas al estado real, desplegable como sitio estático (GitHub Pages/Vercel), cualquiera lo prueba sin instalar nada, ideal portfolio | Hay que reescribir el motor en TS; modelos grandes (difusión, GPT real) limitados a lo que cabe en navegador |
| B. Backend Python (FastAPI + WebSocket) + front                                     | Reutiliza NumPy/PyTorch, modelos más grandes                                                                                                                               | Latencia, hay que hostear un servidor (coste, caídas), peor como demo pública, `npm run dev` no basta        |
| C. Híbrido (A por defecto + backend opcional para modelos pesados)                  | Lo mejor de ambos                                                                                                                                                          | Más complejidad; riesgo de dispersión                                                                        |

**Recomendación: A**, con la puerta abierta a C más adelante (modelos preentrenados exportados a ONNX/ONNX Runtime Web para transformers/difusión, como hace Transformer Explainer).

**Matiz aportado por el usuario:** no se prevé hosting propio; se distribuye vía GitHub y la instalación debe ser trivial para terceros ("se ejecuta en el ordenador del usuario"). Esto _refuerza_ A: solo requiere Node (`git clone && npm install && npm run dev`); una demo estática gratuita en GitHub Pages es posible solo con A (con B habría que alojar Python y no hay opción gratuita fiable). Python quedaría solo como _tooling offline del autor_ (entrenar y exportar mini-modelos y pesos que se versionan en el repo), nunca como requisito del usuario. **PENDIENTE de confirmación final.**

### D2 — ¿Qué pasa con el código Python actual?

| Opción                                                                                                                         | Pros                                                               | Contras                                                 |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------- |
| **A. Mover a `reference/python/` como oráculo de tests** (generar _golden fixtures_; el motor TS debe coincidir numéricamente) | Demuestra rigor (verificación cruzada), conserva el trabajo previo | Mantener dos implementaciones                           |
| B. Borrarlo                                                                                                                    | Repo más simple                                                    | Se pierde historia técnica y el oráculo de verificación |

**Recomendación: A.** Streamlit se elimina (`app.py`, `visuals.py`).

**DECIDIDO (usuario):** se mueve temporalmente a `reference/` y se borrará cuando termine todo el trabajo. Para que el borrado final no rompa nada, los _golden fixtures_ (JSON) se generan una vez y se versionan; los tests de paridad del motor TS leen el JSON, no ejecutan Python.

**Decididos también:** D3 = React + Vite + TS + Tailwind (+ Framer Motion, Zustand, Canvas/SVG, react-three-fiber). Alcance = _todos_ los modelos; primer módulo tras el MLP v2 = Transformer/LLM, pero la navegación de la app seguirá un orden pedagógico (Foundations → CNN → RNN → Transformer → generativos…), independiente del orden de desarrollo.

### D3 — Stack del frontend

Propuesta: **Vite + React 19 + TypeScript (strict) + Tailwind CSS v4 + Framer Motion + Zustand**, render con **SVG/Canvas 2D** (redes, fronteras de decisión, gráficas; D3 solo para escalas/utilidades) y **Three.js / react-three-fiber** para vistas 3D (estilo bbycroft LLM). Tests: **Vitest + Playwright**; calidad: ESLint + Prettier + `tsc`; CI: GitHub Actions.

| Alternativa                  | Pros                                                                                    | Contras                                                  |
| ---------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **React + Vite (propuesta)** | Ecosistema enorme para animación/3D/UI, valorado en portfolios, `npm run dev` inmediato | Más "boilerplate" que Svelte                             |
| Svelte/SvelteKit             | Animaciones y estado muy ergonómicos, bundles pequeños                                  | Menos ecosistema 3D/UI, menos reconocible en CV          |
| Next.js                      | SSR/SEO                                                                                 | SSR innecesario para una app 100 % cliente; más complejo |

### D4 — Motor de ML en el navegador

- **Motor propio en TypeScript** (tensores `Float32Array`, autograd mínimo, capas, optimizadores): coherente con el eslogan _"built from scratch"_ y permite exponer cada activación/gradiente para visualizarlos. Corre en Web Worker (UI nunca se bloquea).
- Para modelos pesados (transformer real, difusión, CNN grandes): **ONNX Runtime Web / WebGPU** con pesos preentrenados pequeños, mostrando los tensores internos.
- Alternativa descartada como base: TF.js (caja negra, poco control de internals, proyecto en mantenimiento).

### D5 — Identidad visual (propuesta, afinar con referencias)

Dark-first (con tema claro), estética "instrumento científico premium": fondo casi negro con malla/ruido sutil, acentos neón (cian/violeta/ámbar) con significado semántico fijo (positivo/negativo, train/test, clases), tipografía Space Grotesk/Inter + JetBrains Mono para números, glassmorphism contenido (no abusar), micro-interacciones y animaciones con propósito (pulsos de datos en el forward, flujo inverso en el backward, grosor/color de pesos vivos, boundary morphing suave). `prefers-reduced-motion` respetado.

## 4. Arquitectura propuesta

```
neural-network-playground/
├─ apps/web/                  # Vite + React app
│  └─ src/
│     ├─ app/                 # shell, router, layout, command palette
│     ├─ design-system/       # tokens, componentes UI, motion presets
│     ├─ modules/             # un módulo por familia de modelos (lazy-loaded)
│     │   ├─ mlp-playground/
│     │   ├─ cnn/  transformer/  gan/  diffusion/ ...
│     │   └─ <module>/{engine,viz,lessons,store}
│     └─ shared/              # hooks, workers, utilidades de render
├─ packages/engine/           # motor ML TS puro (sin DOM), 100 % testeable
├─ reference/python/          # NumPy original + generador de golden fixtures
├─ docs/                      # arquitectura, guía para añadir módulos, ADRs
└─ .github/workflows/         # CI: lint + typecheck + unit + e2e + deploy
```

Principios: módulos plug-in con contrato común (`ModuleDefinition`: metadata, ruta, lecciones, componente) para que **añadir un modelo nuevo = añadir una carpeta**; code-splitting por módulo; motor desacoplado de la UI; estado serializable en la URL (compartir configuraciones, como el playground de TensorFlow).

## 5. Fases

### Fase 0 — Fundamentos (1 sesión)

- Cerrar D1–D5. Scaffold monorepo (pnpm/npm workspaces), TS strict, ESLint/Prettier, Vitest, Playwright, GitHub Actions.
- Design system: tokens (color, espaciado, tipografía, elevación, motion), componentes base (Button, Slider, Toggle, Tabs, Tooltip, Card, Popover, Kbd, Badge), tema oscuro/claro.
- Mover Python a `reference/python/`.
- **Aceptación:** `npm install && npm run dev` abre una landing vacía pero ya hermosa; CI verde.

### Fase 1 — Motor ML en TypeScript (1–2 sesiones)

- Tensores, capas Dense, activaciones (relu, leaky, tanh, sigmoid, elu, swish, linear, softmax), pérdidas (CCE, MSE, MAE, Huber), L1/L2, skip connections, init He/Xavier, optimizadores (SGD, Momentum, RMSprop, Adam), schedulers (constant, exp decay, cosine).
- RNG con semilla (reproducibilidad, `seed` en la URL). Datasets 2D (circle, xor, spiral, gaussian, moons, checkerboard, regresión...).
- Ejecución en Web Worker con protocolo de mensajes tipado (play/pause/step/reset, snapshots de pesos/activaciones/gradientes).
- **Aceptación:** tests de paridad numérica contra los fixtures de Python (tolerancia 1e-6) + test de gradient checking; benchmark documentado.

### Fase 2 — MLP Playground v2 (la pieza estrella, 2–3 sesiones)

Reemplazo del dashboard actual y superación del playground de TensorFlow:

- Red como grafo interactivo: añadir/quitar capas y neuronas con clic/arrastre, **hover en neurona → mini-mapa de decisión de esa neurona**, hover en conexión → peso, gradiente y fórmula; ancho/color de aristas animados por peso.
- Modo **"Step"**: avanzar forward y backward paso a paso con partículas de datos y gradientes fluyendo, panel con la matemática del paso actual (LaTeX/KaTeX).
- Frontera de decisión en Canvas/WebGL con interpolación suave; puntos train/test arrastrables (añadir/mover datos "pintando").
- Curvas loss/accuracy, paisaje de pérdida 2D/3D con trayectoria del optimizador, histogramas de pesos/gradientes (vanishing/exploding visibles).
- Configuración compartible por URL, presets ("Spiral challenge", "Overfit demo"…), tour guiado inicial.
- **Aceptación:** 60 fps con redes típicas, sin parpadeos, responsive, accesible por teclado, Lighthouse ≥ 90.

### Fase 3 — Shell de producto y modo aprendizaje (1–2 sesiones)

- Landing con hero animado (red viva de fondo), galería de módulos con previews animadas, navegación, command palette (⌘K), buscador, glosario con tooltips globales.
- "Learn mode": lecciones por módulo con pasos anclados a la visualización (scrollytelling), niveles beginner/intermediate/advanced, quizzes ligeros.
- Capturas/GIF de portfolio, README nuevo, deploy a GitHub Pages/Vercel, SEO/OG image.

### Fase 4+ — Biblioteca de modelos (iterativa; cada módulo = una mini-entrega con el mismo estándar visual)

Orden propuesto por impacto/valor didáctico:

1. **Foundations lab:** perceptrón, neurona única, funciones de activación, gradient descent en paisajes, optimizadores comparados lado a lado, regularización/overfitting, backprop interactivo en grafo computacional.
2. **CNN:** convoluciones con kernel editable, feature maps por capa, pooling, MNIST en vivo (dibuja un dígito), Grad-CAM, filtros aprendidos.
3. **Transformer / LLM:** tokenización, embeddings, atención multi-cabeza (hover token → pesos), positional encoding, softmax con temperatura/top-k/top-p, mini-GPT en navegador (estilo Transformer Explainer) y **vista 3D por capas** de un LLM (estilo bbycroft).
4. **RNN / LSTM / GRU:** desenrollado temporal, puertas visibles, vanishing gradient en el tiempo.
5. **Autoencoders / VAE:** espacio latente navegable (arrastra un punto → se genera la salida).
6. **GANs:** generador vs discriminador en 2D en vivo, mode collapse visible.
7. **Diffusion:** proceso forward de ruido y reverse denoising paso a paso (2D toy → imágenes pequeñas).
8. **Embeddings / NLP clásico:** word2vec, similitud, analogías, proyección UMAP/t-SNE interactiva.
9. **ML clásico para contexto:** k-NN, SVM (kernels), árboles/random forests, k-means, regresión lineal/logística, PCA.
10. **RL:** Q-learning en gridworld, DQN/policy gradient en CartPole.
11. **Otros:** GNNs, mixture-of-experts, LoRA/fine-tuning, KAN, attention variants, ViT, Neural ODEs, SSM/Mamba.

### Transversal en todas las fases

- Accesibilidad (WCAG AA, teclado, ARIA, `prefers-reduced-motion`), rendimiento (presupuesto de bundle por módulo, workers), i18n-ready (app en inglés), tests (unit del motor, e2e de flujos clave, visual regression opcional), documentación y ADRs, guía "cómo añadir un módulo".

## 6. Riesgos y mitigaciones

| Riesgo                              | Mitigación                                                                                   |
| ----------------------------------- | -------------------------------------------------------------------------------------------- |
| Alcance infinito ("suite completa") | Fases cerradas con criterios de aceptación; cada módulo es independiente y publicable        |
| Visuales bonitos pero lentos        | Presupuesto de rendimiento + Canvas/WebGL para lo denso, SVG solo para lo interactivo/ligero |
| Errores matemáticos silenciosos     | Gradient checking, paridad con oráculo Python, tests de propiedades                          |
| Modelos pesados en navegador        | ONNX/WebGPU con modelos pequeños, carga perezosa, fallback explicado                         |
| Inconsistencia visual entre módulos | Design system + tokens semánticos de color + revisiones visuales por fase                    |

## 7. Flujo de trabajo y Git

- No se hace ningún commit/push desde aquí. Al terminar cada tarea/fase se actualiza **`COMMIT_PLAN.md`** con commits atómicos propuestos (Conventional Commits): mensaje, archivos incluidos y orden, para que los ejecutes tú.
- Se verifica cada entrega ejecutándola de verdad en el navegador (capturas incluidas) antes de darla por buena.

## 8. Preguntas abiertas

1. D1–D5 (arriba): ¿confirmas las recomendaciones?
2. ¿Gestor de paquetes (npm / pnpm)? ¿Hosting objetivo (GitHub Pages / Vercel)?
3. ¿Nombre/branding de la app (¿mantener "Neural Network Playground"?) y paleta: ¿te encaja el neón cian/violeta sobre negro, o prefieres otra dirección?
4. ¿Prioridad real tras el MLP: transformer/LLM primero (más "wow" para portfolio) o foundations/CNN?
5. ¿Alguna referencia visual concreta (de los enlaces) cuyo _look_ quieras replicar más que su contenido?

---

## Estado de ejecución

- ✅ **Fase 0 completada** (monorepo, design system claro/oscuro, landing animada, CI, tests unitarios + e2e). Decisiones D1–D5 cerradas: navegador, Python → `reference/` (temporal), React + Vite + TS + Tailwind, motor propio en TS, paleta Parchment/Obsidian con selector claro/oscuro/sistema, npm.
- ✅ **Fase 1 completada** (motor TS: matrices, 8 activaciones, 4 pérdidas, 4 optimizadores, schedulers, L1/L2, skip connections, 8 datasets, 12 features, `PlaygroundSession` + protocolo de Web Worker). 24 casos de paridad con NumPy (1e-8), gradient checking, benchmark documentado en `docs/ARCHITECTURE.md`.
- ✅ **Fase 2 completada** (MLP Playground v2): grafo interactivo con mini-mapas por neurona, plano de decisión con sondeo al pasar el ratón, curvas (log), edición en vivo de arquitectura/hiperparámetros, 10 experimentos guiados (con lecciones verificadas por tests), URL compartible, **inspector de un paso** (forward → pérdida → backward → actualización con números reales y matemáticas), worker, tema claro/oscuro, responsive. 66 tests web + 109 del motor + 16 e2e.
- ⏭️ **Siguiente:** Fase 2.5 opcional (paisaje de pérdida, arrastrar puntos de datos, histogramas) y **Fase 3** (shell de producto: landing final, modo aprendizaje, despliegue, pasada de rendimiento) o saltar al primer módulo nuevo (Transformer/LLM).

- ✅ **Todos los módulos del camino de aprendizaje están construidos** (en orden pedagógico): Foundations (neurona, activaciones, descenso por gradiente), MLP, CNN, RNN/LSTM/GRU, Transformer/LLM (con vista 3D), Autoencoders/VAE, GANs, Difusión, Embeddings/NLP (word2vec) y RL tabular. Cada uno tiene motor propio en TypeScript con tests (gradient checks, paridad o referencia exacta), experimentos guiados cuyas afirmaciones verifica la suite, e2e y capturas. Pendiente: Fase 3 (despliegue, modo aprendizaje, pasada de rendimiento), Fase 2.5 opcional y borrar `reference/` cuando decidas que ya no hace falta.

- ✅ **Módulo Transformer/LLM (v1, explorador 2D)**: GPT diminuto propio (2 capas, 4 cabezas, d=64, 136k parámetros) entrenado desde cero en NumPy con retropropagación manual sobre un corpus sintético gramatical; tokenizador BPE a nivel de bytes propio; forward en TypeScript con paridad 1e-8 frente a NumPy; UI con tokens + demo BPE, embeddings (+ mapa PCA), atención por cabeza con matemáticas al pasar el ratón, corriente residual y MLP (GELU), distribución del siguiente token con temperatura/top-k/top-p, generación determinista, y 10 experimentos guiados cuyas afirmaciones verifican los tests. Pendiente para el módulo: vista 3D por capas (estilo bbycroft), carga opcional de GPT-2.
