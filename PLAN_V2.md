# PLAN V2: de «Neural Network Playground» a un atlas visual de Machine Learning

> Estado: **EN EJECUCIÓN.** Decisiones cerradas por el usuario (todas = recomendada): nombre
> `see-inside-ml`; 3 paquetes `core`/`classic`/`deep`; orden = clásico primero con lotes rápidos de Deep
> intercalados; datos = sintéticos + Iris/Wine/Diabetes/Breast Cancer diminutos con atribución.
> Fase 0 completada (renombrado, paquetes, registro por pistas, lab-kit, primer laboratorio clásico,
> workflow de Pages). Sigue en vigor la regla de siempre:
> **Estado detallado y lista de lo pendiente: [docs/ROADMAP.md](docs/ROADMAP.md).**
> **Progreso (oct. 2026):** hechas las fases 0–5, el lote rápido de Deep 1 y 2, y de la fase 6–8 los labs
> toolbox, charlm, compression, lora, explain, bandits, policy-gradient, td-learning, mcts, dyna y
> position encodings, KV cache, aritmética de convoluciones, tokenizadores, n-gramas vs LM neuronal scaling laws, actor-critic, DQN, augmentation, head ablation, variantes de VAE, GAN, difusión, flows y atribución de píxeles y robustez adversaria , probing y visualización de características y mixture of experts, GNN, neural ODE, RAG y RLHF, análisis de supervivencia y optimización con restricciones (89 labs en total). Pendiente:
> extras clásicos opcionales y la fase 9 (endurecimiento).
> no se hace ningún commit ni push; al final de cada fase se actualiza `COMMIT_PLAN.md`.

## 0. Visión

Hasta ahora: diez laboratorios de Deep Learning, cada uno con motor propio en TypeScript, experimentos
cuyas afirmaciones verifica la suite de tests, y UI interactiva. El nuevo objetivo:

> **Ver el interior de cualquier modelo de Machine Learning**, clásico o profundo: cómo se ajusta,
> qué decide, por qué falla. Todo en el navegador, todo escrito desde cero y verificado.

Tres ejes de ampliación (el usuario descartó por ahora «pedagogía y producto»: modo curso, comparador
genérico, etc.):

1. **Profundizar** en los módulos de Deep Learning que ya existen (adversariales, Grad-CAM, DQN, GPT que se
   entrena en el navegador, difusión sobre dígitos…).
2. **Módulos nuevos**: optimización, generalización, interpretabilidad, temas actuales (MoE, LoRA, cuantización…).
3. **Machine Learning clásico completo**: regresión, regresión logística, árboles, random forests,
   boosting, SVM, procesos gaussianos, clustering, reducción de dimensión, modelos probabilísticos,
   series temporales… «todos los que podamos».

El tercer eje cambia la naturaleza del proyecto (ya no es solo «neural network»), de ahí el cambio de
nombre y una reestructuración.

## 1. Decisiones a cerrar (con pros y contras)

### D1. Nombre del proyecto y del repositorio

| Opción                           | Pros                                                                                      | Contras                                                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `see-inside-machine` (propuesta) | Transmite la idea de «ver dentro»; ya la tienes en mente                                  | En inglés suena incompleto («see inside _the_ machine»); «machine» sin «learning» es ambiguo en búsquedas |
| **`see-inside-ml`**              | Misma idea, correcta gramaticalmente, corto, acota el tema, buen slug (`/see-inside-ml/`) | «ML» menos poético                                                                                        |
| `inside-the-machine`             | Natural y evocador («the ghost in the machine»), buena marca para portfolio               | Poco descriptivo; compite con muchos resultados no relacionados                                           |
| `ml-atlas`                       | Sugiere catálogo/mapa de modelos, corto                                                   | Genérico; hay proyectos con ese nombre                                                                    |
| `glassbox-ml`                    | «Caja de cristal» frente a caja negra                                                     | Colisión con _Glassbox/InterpretML_ (término establecido)                                                 |

Recomendación: **`see-inside-ml`** (lema: _«See inside every machine-learning model»_). Si prefieres un
nombre más de marca: `inside-the-machine`. Mecánica del cambio (la hago yo): renombrar el repo en GitHub
(redirige solo), `REPO_URL`, paquetes `@nnp/*` → `@sim/*`, `Wordmark`, claves de `localStorage`
(`nnp-theme` → migración), título, README, CI y base de GitHub Pages (`/see-inside-ml/`).
**Como no hay commits todavía, renombrar y reestructurar ahora no cuesta nada de historial.**

### D2. Estructura del repositorio

Hoy: un único paquete `packages/engine` (≈8 000 líneas, 10 módulos). Con ~30 algoritmos clásicos más
las ampliaciones pasaría de 40 000 líneas en un solo paquete.

| Opción                                          | Pros                                                                                                                       | Contras                                                       |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| A. Un solo `engine`                             | Cero trabajo de migración                                                                                                  | Acoplamiento, tests cada vez más lentos, sin fronteras claras |
| **B. Tres paquetes: `core`, `classic`, `deep`** | Fronteras claras; tests por paquete en paralelo; `classic` no arrastra nada de Deep Learning; reutilizable fuera de la web | Un día de migración de imports y configs                      |
| C. Un paquete por laboratorio                   | Aislamiento máximo                                                                                                         | Excesivo: 100+ paquetes, config y CI inmanejables             |

Recomendación: **B**. Propuesta de árbol:

```
see-inside-ml/
├─ apps/web/
│  └─ src/
│     ├─ app/            shell, registro de laboratorios, navegación por pistas
│     ├─ shared/         canvas, gráficos, lab-kit (ver §2.3)
│     └─ labs/<pista>/<laboratorio>/     (antes: modules/<modulo>)
├─ packages/
│  ├─ core/              rng, matrices, álgebra lineal, métricas, datasets, optimizadores, interfaces
│  ├─ classic/           regresión, árboles, SVM, clustering, probabilístico, series temporales…
│  └─ deep/              mlp, cnn, rnn, transformer, generativos, rl, lenguaje, interpretabilidad
├─ tools/                herramientas de autoría en Python (fixtures sklearn, entrenar el GPT…)
└─ docs/
```

### D3. Cómo se valida el ML clásico

Tengo `scikit-learn 1.9`, `scipy`, `statsmodels` y `xgboost` instalados, así que puedo repetir lo que hice con
NumPy en el MLP: **generar fixtures con la librería de referencia** (herramienta de autoría, no se usa en
tiempo de ejecución) y comprobar paridad.

| Opción                                                                   | Pros                                                         | Contras                                                         |
| ------------------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------------- |
| **Paridad con sklearn/scipy/statsmodels/xgboost + tests de propiedades** | Evidencia fuerte de corrección; es un argumento de portfolio | Hay que ajustar tolerancias y fijar desempates/inicializaciones |
| Solo tests analíticos y de propiedades                                   | Sin dependencia de Python                                    | Menos convincente; fácil colar errores sutiles                  |

Recomendación: **paridad siempre que el algoritmo sea determinista** (OLS, ridge, lasso, logística, LDA/QDA,
Naive Bayes, k-NN, CART con desempate fijo, PCA, linkage jerárquico, DBSCAN, GP, ARIMA…), y
**propiedades/soluciones conocidas** para los estocásticos o aproximados (t-SNE, UMAP, MCMC, k-means con
semilla, bosques: error OOB, tests de Kolmogorov–Smirnov, trustworthiness…).

### D4. Datos

| Opción                                                                                                                                  | Pros                                                                                    | Contras                                       |
| --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------- |
| Solo sintéticos (procedurales, como hasta ahora)                                                                                        | Sin licencias, sin descargas, reproducibles, controlables (outliers, ruido, desbalance) | Menos «reales»                                |
| **Sintéticos + un puñado de clásicos diminutos incrustados** (Iris, Wine, Diabetes, Breast Cancer: dominio público o BSD/CC-BY, <50 kB) | Autenticidad (la gente reconoce Iris); permite comparar con sklearn de verdad           | Hay que citar licencias; ligeramente más peso |

Recomendación: **sintéticos como base** (2D editables: blobs, lunas, círculos, espirales, XOR, desbalanceado,
curvas de regresión, heterocedasticidad, outliers…) **+ 4–5 datasets clásicos** diminutos con atribución.
Subir un CSV propio queda como extra opcional al final (toca «producto»).

### D5. Orden de ejecución

| Opción                                                    | Pros                                                                                                                                               | Contras                                                                 |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Clásico primero, Deep después**                         | La infraestructura nueva (lab-kit, datasets, paridad) se construye una vez y se aprovecha en todo; el ML clásico es la base conceptual de lo demás | Los avances de Deep que más lucen (adversariales, DQN) llegan más tarde |
| Deep primero, clásico después                             | Resultados vistosos antes                                                                                                                          | La restructuración y el lab-kit se harían con prisas o dos veces        |
| Intercalado (una fase clásica + un «lote rápido» de Deep) | Variedad y ritmo                                                                                                                                   | Más cambio de contexto                                                  |

Recomendación: **clásico primero**, con dos «lotes rápidos» de Deep intercalados (ejemplos adversariales +
Grad-CAM; bandidos y REINFORCE) para no dejar el Deep parado.

### D6. Despliegue

Propongo **desplegar tras la Fase 0** (GitHub Pages con CI) y publicar cada fase: URL pública desde el
principio, y cada fase se ve en producción. Contra: hay que cuidar el `base` de las rutas (HashRouter ya lo
resuelve) y el tamaño por ruta (cada laboratorio ya se carga bajo demanda).

## 2. Arquitectura propuesta

### 2.1 Registro por pistas

`registry.ts` pasa de lista plana a **pistas** (tracks) con laboratorios ordenados pedagógicamente:

1. **Fundamentos** · 2. **ML clásico supervisado** · 3. **ML clásico no supervisado y probabilístico** ·
2. **Evaluación y optimización** · 5. **Deep Learning** · 6. **Generativos** · 7. **Lenguaje** ·
3. **Aprendizaje por refuerzo** · 9. **Interpretabilidad y robustez** · 10. **Temas actuales**.

Cada entrada: `id`, `track`, `title`, `tagline`, `level`, `status`, `topics`, `prerequisites`, `path`.
La navegación pasa del menú plano «Labs» a un **menú por pistas** (con 100 laboratorios es imprescindible);
la home muestra las pistas como tarjetas y el camino de cada una.

### 2.2 Interfaces comunes en `core` (la clave para llegar a «todos los modelos»)

```ts
interface Classifier {
  fit(X, y, params);
  predict(X);
  predictProba?(X);
}
interface Regressor {
  fit(X, y, params);
  predict(X);
}
interface Clusterer {
  fit(X, params);
  labels;
  predict?(X);
}
interface Projector {
  fit(X, params);
  transform(X);
  inverse?(Z);
}
interface Iterative<S> {
  states(): Generator<S>;
} // para animar: nodos de un árbol, rondas de boosting, iteraciones EM, pasos SMO
```

Con ellas cada algoritmo es **un fichero** + sus fixtures, y los tests genéricos se escriben una vez:
determinismo con semilla, mejora sobre el azar, tolerancia a datos degenerados, `predict` con la forma
correcta, no mutar la entrada.

### 2.3 `lab-kit` en `apps/web/src/shared` (UI reutilizable)

- `DataPlane`: lienzo 2D con **regiones de decisión/regresión genéricas** (`predictGrid`), edición de datos
  (añadir, arrastrar, cambiar clase), puntos de test, zoom.
- `ParamPanel`: controles generados desde un **esquema declarativo** (sliders, selects, interruptores con
  nota, formato, «aplica en vivo» frente a «reinicia»).
- `MetricsStrip` (train/test, matriz de confusión, R², RMSE…), `StepPlayer` (reproducir/pausar/avanzar/arrastrar
  por los estados de un `Iterative`), `CoefficientPath`, `TreeView`, `Dendrogram`, `ScatterMatrix` (pares de
  variables + proyección PCA para más de 2 variables), `ResidualPlot`, `ROCPlot`.
- `useWorkerJob`: arnés genérico para tareas pesadas (t-SNE, UMAP, GP, MCMC, boosting largo) con progreso y
  cancelación, sobre el protocolo de workers que ya tenemos.
- Cada laboratorio = configuración + paneles propios (por ejemplo `TreeView` en árboles). Así hay
  consistencia sin que todo parezca igual.

### 2.4 Definición de «hecho» de cada laboratorio (igual que ahora)

Motor con tests (+ paridad cuando aplica) · UI en el navegador, claro/oscuro, móvil sin desbordes · 3–6
experimentos guiados **cuyas afirmaciones se miden y se verifican en tests** (explorar primero, afirmar
después) · spec e2e · README/ARCHITECTURE · captura · entrada en `COMMIT_PLAN.md`.

## 3. Catálogo completo (y qué se ve en cada uno)

Leyenda de tamaño: **S** ≈ pequeño · **M** ≈ medio (como un lab actual) · **L** ≈ grande.
Nivel de prioridad: **A** (núcleo, imprescindible) · **B** (ampliado) · **C** (extra si da tiempo).

### Pista 2 · ML clásico supervisado

| Lab                                                   | Qué enseña / qué se ve                                                                                                                                        | Validación                    | Tam. | Prio. |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---- | ----- |
| Regresión lineal simple                               | Mínimos cuadrados (áreas de los cuadrados de error), residuos, R², fórmula cerrada vs descenso por gradiente, outliers y apalancamiento, cuarteto de Anscombe | sklearn/statsmodels           | M    | A     |
| Regresión lineal múltiple                             | Ecuaciones normales, estandarización, coeficientes con intervalos, multicolinealidad (VIF), interpretación                                                    | statsmodels (SE, p-valores)   | M    | A     |
| Polinomial y regularización                           | Ridge, Lasso, Elastic Net: trayectorias de coeficientes, sobreajuste, sesgo-varianza                                                                          | sklearn (ridge/lasso exactos) | M    | A     |
| Regresión robusta y GLM                               | Huber, RANSAC, cuantílica, Poisson/Gamma                                                                                                                      | sklearn/statsmodels           | M    | B     |
| Regresión logística                                   | Frontera, odds, softmax multiclase, calibración, regularización, IRLS vs GD                                                                                   | sklearn                       | M    | A     |
| k-NN                                                  | k, métricas, ponderación, Voronoi, maldición de la dimensión                                                                                                  | sklearn                       | M    | A     |
| Naive Bayes, LDA y QDA                                | Clasificadores generativos, elipses gaussianas, regularización (shrinkage)                                                                                    | sklearn                       | M    | A     |
| Árboles de decisión (CART)                            | Crecimiento paso a paso, Gini/entropía/MSE, regiones y vista del árbol, poda por complejidad                                                                  | sklearn                       | L    | A     |
| Random forest                                         | Bagging, error OOB, importancia de variables, `mtry`, decorrelación                                                                                           | sklearn (propiedades + OOB)   | M    | A     |
| AdaBoost                                              | Pesos de los puntos que cambian ronda a ronda                                                                                                                 | sklearn                       | M    | A     |
| Gradient boosting y estilo XGBoost                    | Ajuste de residuos, _shrinkage_, 2.º orden, regularización, regresión y clasificación                                                                         | sklearn / xgboost             | L    | A     |
| SVM                                                   | Margen, vectores soporte, C, kernels (lineal, RBF, polinomial), truco del kernel con elevación 3D, SMO paso a paso                                            | sklearn (libsvm)              | L    | A     |
| Procesos gaussianos y kernel ridge                    | Kernels, bandas de incertidumbre, verosimilitud marginal                                                                                                      | sklearn                       | L    | A     |
| Regresión bayesiana                                   | Distribución posterior de las rectas, incertidumbre predictiva                                                                                                | solución analítica            | M    | B     |
| Perceptrón multicapa vs clásicos («Duelo de modelos») | Todos los clasificadores sobre el **mismo** dataset, frontera y métricas lado a lado                                                                          | los de arriba                 | M    | A     |
| Ensembles: votación y stacking                        | Combinar modelos distintos y cuándo ayuda                                                                                                                     | propiedades                   | S    | B     |
| Regresión isotónica, splines/GAM, MARS                | Modelos flexibles interpretables                                                                                                                              | sklearn/pygam                 | M    | C     |

### Pista 3 · ML clásico no supervisado y probabilístico

| Lab                                   | Qué se ve                                                           | Validación                    | Tam. | Prio. |
| ------------------------------------- | ------------------------------------------------------------------- | ----------------------------- | ---- | ----- |
| k-means y k-means++                   | Iteraciones de Lloyd animadas, codo, silueta, inicialización        | sklearn (con inicio dado)     | M    | A     |
| Clustering jerárquico                 | Enlaces (single/complete/average/Ward), dendrograma y corte         | scipy `linkage` exacto        | M    | A     |
| DBSCAN, OPTICS y mean shift           | Densidad, ruido, formas no convexas                                 | sklearn exacto                | M    | A     |
| Mezclas gaussianas y EM               | Elipses y responsabilidades por iteración, BIC/AIC                  | sklearn (inicio dado)         | L    | A     |
| Clustering espectral                  | Laplaciano del grafo, autovectores                                  | sklearn                       | M    | B     |
| PCA y kernel PCA                      | Ejes, varianza explicada, reconstrucción, blanqueo                  | sklearn exacto                | M    | A     |
| Aprendizaje de variedades             | MDS, Isomap, LLE, **t-SNE** (perplejidad), **UMAP**                 | propiedades (trustworthiness) | L    | A     |
| ICA y NMF                             | Separación ciega de señales (cóctel), temas                         | sklearn                       | M    | B     |
| Detección de anomalías                | Isolation Forest, LOF, one-class SVM, Mahalanobis                   | sklearn / propiedades         | M    | A     |
| Mapas autoorganizados (SOM)           | Cuadrícula de Kohonen deformándose                                  | propiedades                   | S    | B     |
| Reglas de asociación y recomendadores | Apriori, filtrado colaborativo, factorización de matrices (ALS/SGD) | exactos / propiedades         | M    | B     |
| Modelos ocultos de Markov             | Forward-backward, Viterbi, Baum–Welch                               | enumeración exhaustiva        | L    | A     |
| Filtro de Kalman y de partículas      | Seguimiento con ruido                                               | solución cerrada              | M    | B     |
| MCMC e inferencia bayesiana           | Metropolis–Hastings, Gibbs, HMC, a priori conjugados, posteriores   | KS contra posterior conocido  | L    | A     |
| Cadenas de Markov y PageRank          | Distribución estacionaria, caminatas aleatorias                     | exacto                        | S    | B     |
| Series temporales                     | Suavizado, descomposición, ARIMA, estacionariedad, previsión        | statsmodels                   | L    | A     |
| Análisis de supervivencia             | Kaplan–Meier, Cox                                                   | lifelines/statsmodels         | M    | C     |

### Pista 4 · Evaluación y optimización

| Lab                                 | Qué se ve                                                                                           | Tam. | Prio. |
| ----------------------------------- | --------------------------------------------------------------------------------------------------- | ---- | ----- |
| Sesgo, varianza y generalización    | Curvas de aprendizaje, sobreajuste, _double descent_                                                | M    | A     |
| Validación                          | Train/val/test, k-fold, fuga de datos, validación anidada                                           | M    | A     |
| Métricas y umbrales                 | Confusión, ROC, PR, calibración, desbalance (pesos, SMOTE), coste                                   | M    | A     |
| Ajuste de hiperparámetros           | Rejilla, aleatoria, _successive halving_, **optimización bayesiana** (GP + EI)                      | L    | A     |
| Selección e ingeniería de variables | Filtro/envoltura/embebido, importancia por permutación                                              | M    | B     |
| Carrera de optimizadores            | Rosenbrock, Himmelblau, SGD/momentum/Nesterov/Adam/L-BFGS/Newton, planificadores, tamaño de batch   | M    | A     |
| Metaheurísticos                     | Algoritmo genético, recocido simulado, enjambre de partículas, CMA-ES                               | L    | B     |
| Optimización con restricciones      | Programación lineal (símplex), cuadrática, multiplicadores de Lagrange (conecta con el dual de SVM) | L    | C     |

### Pista 5 · Deep Learning (profundizar en lo que ya existe)

| Módulo            | Ampliaciones                                                                                                                                                                                | Tam. | Prio. |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ----- |
| Fundamentos / MLP | Paisaje de pérdida 2D, arrastrar puntos, histogramas de pesos/gradientes, dropout, batch norm, comparativa de inicializaciones, _weight decay_, _early stopping_, buscador de learning rate | L    | A     |
| CNN               | **Ejemplos adversariales (FGSM/PGD)**, Grad-CAM, aumento de datos, aritmética de convoluciones (padding/stride), campo receptivo, visualización de filtros                                  | L    | A     |
| RNN               | Modelo de lenguaje por caracteres que genera texto, seq2seq con atención (copiar/invertir), bidireccional                                                                                   | M    | A     |
| Transformer       | **Entrenar el GPT en el navegador**, caché KV, codificaciones posicionales (sinusoidal/RoPE/ALiBi), ablación de cabezas, cargador opcional de un modelo real pequeño                        | L    | A     |
| Autoencoder/VAE   | β-VAE y desentrelazado, VAE condicional, VQ-VAE                                                                                                                                             | M    | B     |

### Pista 6 · Generativos

GAN sobre dígitos (imágenes 16×16), cGAN, WGAN-GP; difusión sobre dígitos con **classifier-free guidance** y
_flow matching_; _normalizing flows_ (RealNVP 2D); modelos basados en energía, Hopfield y máquinas de Boltzmann
(C). Tamaño L en conjunto, prioridad A para GAN/difusión sobre dígitos, B el resto.

### Pista 7 · Lenguaje

Comparador de tokenizadores (BPE / WordPiece / Unigram), n-gramas → modelo neuronal y perplejidad, RAG de
juguete (BM25 + embeddings), RLHF de juguete (modelo de recompensa), leyes de escala (entrenar modelos de
distinto tamaño y ajustar una ley de potencias). M cada uno, prioridad B.

### Pista 8 · Aprendizaje por refuerzo

Bandidos (ε-greedy, UCB, Thompson) · Monte Carlo · TD(λ) y trazas de elegibilidad · Dyna-Q y barridos
priorizados · **REINFORCE con _baseline_** · actor-crítico (A2C) · **DQN** (réplica + red objetivo) en CartPole
y MountainCar · MCTS en tres en raya/conecta-4 · dilema del prisionero iterado (C). Prioridad A: bandidos,
REINFORCE, DQN; resto B.

### Pista 9 · Interpretabilidad y robustez

Explicar modelos tabulares (importancia por permutación, PDP/ICE, **SHAP** exacto en árboles pequeños, LIME) ·
saliencia/Grad-CAM/gradientes integrados · robustez y entrenamiento adversarial · visualización de
características y _probing_ lineal sobre activaciones · equidad (métricas por grupo, C). Prioridad A: tabulares
y robustez; B el resto.

### Pista 10 · Temas actuales

Mixture of Experts (enrutado), LoRA (ajuste de bajo rango), cuantización (int8/int4 y su efecto), poda y
destilación (profesor-alumno), redes de grafos (paso de mensajes, clasificación de nodos), redes residuales y
Neural ODE (C). M cada uno, prioridad B.

**Recuento:** ≈ 30 laboratorios clásicos + 8 de evaluación/optimización + ≈ 12 ampliaciones de los DL actuales

- ≈ 30 módulos nuevos de las pistas 6–10. **≈ 80 entradas**, de las cuales ≈ 45 son prioridad A.

## 4. Plan de ejecución por fases

Cada fase termina en estado verde (`npm run check` + e2e), con `COMMIT_PLAN.md` actualizado y capturas.
Como cada fase es grande, la ejecuto laboratorio a laboratorio y te enseño el resultado entre medias.

| Fase                                       | Contenido                                                                                                                                                                                                                                                                                                                                                                            | Entregable / criterio de aceptación                                                                                                                                                 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0. Cimientos**                           | Renombrar; separar en `core`/`classic`/`deep`; registro por pistas + menú por pistas; **lab-kit** (DataPlane, ParamPanel, StepPlayer, MetricsStrip…); biblioteca de datasets (sintéticos + clásicos); interfaces `Classifier/Regressor/Clusterer/Projector/Iterative`; herramienta de fixtures en Python; CI actualizado; despliegue a GitHub Pages; reescritura de `COMMIT_PLAN.md` | Todo lo existente funciona igual con el nuevo nombre y estructura; un primer lab clásico de prueba (regresión lineal simple) construido sobre el lab-kit demuestra que el kit sirve |
| **1. Regresión y clasificadores simples**  | Lineal simple y múltiple, polinomial + regularización, robusta/GLM, logística, k-NN, Naive Bayes/LDA/QDA                                                                                                                                                                                                                                                                             | 7 labs; paridad con sklearn/statsmodels; experimentos medidos                                                                                                                       |
| **2. Árboles, ensembles, SVM y kernels**   | CART, random forest, AdaBoost, gradient boosting/XGBoost, SVM, GP/kernel ridge, regresión bayesiana, **Duelo de modelos**, votación/stacking                                                                                                                                                                                                                                         | 9 labs; paridad con sklearn/xgboost                                                                                                                                                 |
| **Lote rápido 1 (Deep)**                   | Ejemplos adversariales + Grad-CAM en la CNN; histogramas y paisaje de pérdida del MLP                                                                                                                                                                                                                                                                                                | Ampliaciones de módulos existentes                                                                                                                                                  |
| **3. No supervisado**                      | k-means, jerárquico, DBSCAN/OPTICS/mean shift, GMM/EM, espectral, PCA/kPCA, variedades (MDS/Isomap/LLE/t-SNE/UMAP), ICA/NMF, anomalías, SOM                                                                                                                                                                                                                                          | ≈ 11 labs; scipy/sklearn; `useWorkerJob` para t-SNE/UMAP                                                                                                                            |
| **4. Probabilístico y secuencial**         | HMM, Kalman/partículas, MCMC e inferencia bayesiana, Markov/PageRank, series temporales, asociación y recomendadores                                                                                                                                                                                                                                                                 | ≈ 6 labs; enumeración exhaustiva, statsmodels, KS                                                                                                                                   |
| **5. Evaluación y optimización**           | Sesgo-varianza, validación, métricas/ROC/calibración, ajuste de hiperparámetros (con optimización bayesiana), selección de variables, carrera de optimizadores, metaheurísticos                                                                                                                                                                                                      | 7 labs (+ restricciones si hay tiempo)                                                                                                                                              |
| **Lote rápido 2 (Deep)**                   | Bandidos y REINFORCE (RL); RNN con generación de texto                                                                                                                                                                                                                                                                                                                               | Ampliaciones                                                                                                                                                                        |
| **6. Deep Learning, ampliado**             | MLP (dropout, batch norm, inits, early stopping…), CNN (aumento, aritmética), Transformer (entrenar en el navegador, KV, posicionales, ablación), β-VAE/VAE condicional                                                                                                                                                                                                              | Ampliaciones de módulos existentes                                                                                                                                                  |
| **7. Generativos, lenguaje y RL avanzado** | GAN/cGAN/WGAN y difusión sobre dígitos con CFG, flows, comparador de tokenizadores, n-gramas/perplejidad, A2C, DQN, MCTS, Dyna                                                                                                                                                                                                                                                       | ≈ 12 labs                                                                                                                                                                           |
| **8. Interpretabilidad y temas actuales**  | Explicar tabulares (permutación, PDP, SHAP, LIME), robustez, probing; MoE, LoRA, cuantización, destilación, GNN, RAG/RLHF de juguete, leyes de escala                                                                                                                                                                                                                                | ≈ 12 labs                                                                                                                                                                           |
| **9. Endurecimiento**                      | Pasada de rendimiento (chunks por pista, LazyMotion), auditoría de accesibilidad, regresión visual, documentación final                                                                                                                                                                                                                                                              | Presupuestos de tamaño por ruta y cero violaciones de accesibilidad                                                                                                                 |

**Corte natural:** al terminar la Fase 5 el proyecto cubre ya todo el ML clásico, evaluación y optimización
(≈ 45 laboratorios en total con los 10 actuales), que es lo que más cambia su alcance. Las fases 6–8 son
ampliaciones; podemos reordenarlas o recortarlas según el interés.

## 5. Calidad (no se negocia)

- **Corrección:** paridad con la librería de referencia donde el algoritmo es determinista; gradient checks en
  lo diferenciable; soluciones analíticas (Bellman, posteriores conjugados, ecuaciones normales) y propiedades
  en lo estocástico; semillas fijas.
- **Afirmaciones medidas:** todo texto de experimento («este modelo sobreajusta», «con k=1 el error de
  entrenamiento es 0») se mide primero y se afirma en un test (`lessons.test.ts`), como hasta ahora.
- **Tests genéricos de interfaz:** una batería común se ejecuta sobre cada `Classifier`/`Regressor`/
  `Clusterer`/`Projector`.
- **e2e por laboratorio:** secciones, interacción principal, experimentos, y que quepa en el móvil.
- **Rendimiento:** los algoritmos pesados van a workers; cada laboratorio se carga bajo demanda; presupuesto de
  tamaño por ruta vigilado en CI.
- **Accesibilidad:** controles con etiqueta y teclado; el contenido no depende solo del color.

## 6. Riesgos y mitigaciones

| Riesgo                                                | Mitigación                                                                                                                          |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Alcance enorme (≈ 80 entradas)                        | Prioridades A/B/C, fases con entregable, corte natural tras la Fase 5; el lab-kit hace que un laboratorio clásico sea barato        |
| Que todos los laboratorios clásicos se parezcan       | Plantilla común, pero cada uno con su panel propio (árbol, dendrograma, vectores soporte, trayectoria de coeficientes, elipses EM…) |
| Algoritmos pesados en JS (t-SNE, UMAP, SVM, GP, MCMC) | Tamaños pequeños (≤ 1 000 puntos), workers, avance por pasos animable                                                               |
| Diferencias numéricas con sklearn                     | Fijar desempates e inicializaciones; comparar soluciones/predicciones, no rutas; tolerancias documentadas                           |
| Licencias de datasets                                 | Solo los diminutos de dominio público o BSD/CC-BY con atribución visible                                                            |
| Migración de nombre y estructura                      | Se hace ahora, antes del primer commit; revisar imports, CI y rutas con la suite completa                                           |
| Modelos reales grandes (GPT-2)                        | Opcional y al final; evaluar tamaño, cuantización y carga perezosa antes de decidir                                                 |

## 7. Fuera de alcance (por ahora)

Modo curso/itinerario con progreso, comparador genérico de ejecuciones, internacionalización, PWA/offline, subir
CSV propio, WebGPU. Todo ello es compatible con esta arquitectura y se puede retomar después.
