import { Matrix } from './matrix';

/**
 * Dense linear algebra for the classical models: Cholesky and LU solvers, inverse, determinant,
 * and the eigen-decomposition of symmetric matrices (Jacobi rotations). Matrices are small (tens of
 * columns), so clarity and numerical care matter more than speed.
 */

const EPS = 1e-12;

function assertSquare(a: Matrix, op: string): void {
  if (a.rows !== a.cols)
    throw new RangeError(`${op}: matrix must be square, got ${a.rows}×${a.cols}`);
}

/** Lower-triangular `L` with `L·Lᵀ = A` for a symmetric positive-definite `A`. Throws if `A` is not. */
export function cholesky(a: Matrix): Matrix {
  assertSquare(a, 'cholesky');
  const n = a.rows;
  const l = new Matrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = a.get(i, j);
      for (let k = 0; k < j; k++) sum -= l.get(i, k) * l.get(j, k);
      if (i === j) {
        if (!(sum > 0)) throw new RangeError('cholesky: matrix is not positive definite');
        l.set(i, j, Math.sqrt(sum));
      } else {
        l.set(i, j, sum / l.get(j, j));
      }
    }
  }
  return l;
}

/** Solves `L·Lᵀ·x = b` given the Cholesky factor `L` (works column by column when `b` is a matrix). */
export function choleskySolve(l: Matrix, b: Matrix): Matrix {
  const n = l.rows;
  if (b.rows !== n) throw new RangeError('choleskySolve: shape mismatch');
  const x = new Matrix(n, b.cols);
  for (let c = 0; c < b.cols; c++) {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = b.get(i, c);
      for (let k = 0; k < i; k++) sum -= l.get(i, k) * (y[k] as number);
      y[i] = sum / l.get(i, i);
    }
    for (let i = n - 1; i >= 0; i--) {
      let sum = y[i] as number;
      for (let k = i + 1; k < n; k++) sum -= l.get(k, i) * x.get(k, c);
      x.set(i, c, sum / l.get(i, i));
    }
  }
  return x;
}

/** `log det A` of a symmetric positive-definite matrix from its Cholesky factor. */
export function logDetFromCholesky(l: Matrix): number {
  let s = 0;
  for (let i = 0; i < l.rows; i++) s += Math.log(l.get(i, i));
  return 2 * s;
}

/** LU decomposition with partial pivoting: returns the packed factors, the row permutation and its sign. */
export function luDecompose(a: Matrix): { lu: Matrix; perm: number[]; sign: 1 | -1 } {
  assertSquare(a, 'lu');
  const n = a.rows;
  const lu = a.clone();
  const perm = Array.from({ length: n }, (_, i) => i);
  let sign: 1 | -1 = 1;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(lu.get(r, col)) > Math.abs(lu.get(pivot, col))) pivot = r;
    if (Math.abs(lu.get(pivot, col)) < EPS) throw new RangeError('lu: matrix is singular');
    if (pivot !== col) {
      for (let c = 0; c < n; c++) {
        const t = lu.get(col, c);
        lu.set(col, c, lu.get(pivot, c));
        lu.set(pivot, c, t);
      }
      [perm[col], perm[pivot]] = [perm[pivot] as number, perm[col] as number];
      sign = sign === 1 ? -1 : 1;
    }
    for (let r = col + 1; r < n; r++) {
      const f = lu.get(r, col) / lu.get(col, col);
      lu.set(r, col, f);
      for (let c = col + 1; c < n; c++) lu.set(r, c, lu.get(r, c) - f * lu.get(col, c));
    }
  }
  return { lu, perm, sign };
}

/** Solves `A·x = b` for a general square `A` (b may have several columns). */
export function solve(a: Matrix, b: Matrix): Matrix {
  const { lu, perm } = luDecompose(a);
  const n = a.rows;
  if (b.rows !== n) throw new RangeError('solve: shape mismatch');
  const x = new Matrix(n, b.cols);
  for (let c = 0; c < b.cols; c++) {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = b.get(perm[i] as number, c);
      for (let k = 0; k < i; k++) sum -= lu.get(i, k) * (y[k] as number);
      y[i] = sum;
    }
    for (let i = n - 1; i >= 0; i--) {
      let sum = y[i] as number;
      for (let k = i + 1; k < n; k++) sum -= lu.get(i, k) * x.get(k, c);
      x.set(i, c, sum / lu.get(i, i));
    }
  }
  return x;
}

export function identity(n: number): Matrix {
  const m = new Matrix(n, n);
  for (let i = 0; i < n; i++) m.set(i, i, 1);
  return m;
}

export function inverse(a: Matrix): Matrix {
  return solve(a, identity(a.rows));
}

export function determinant(a: Matrix): number {
  try {
    const { lu, sign } = luDecompose(a);
    let d = sign as number;
    for (let i = 0; i < a.rows; i++) d *= lu.get(i, i);
    return d;
  } catch {
    return 0;
  }
}

export function transpose(a: Matrix): Matrix {
  const t = new Matrix(a.cols, a.rows);
  for (let i = 0; i < a.rows; i++) for (let j = 0; j < a.cols; j++) t.set(j, i, a.get(i, j));
  return t;
}

/** `XᵀX` and `Xᵀy` are everywhere in least squares: computed without forming the transpose. */
export function gram(x: Matrix): Matrix {
  const g = new Matrix(x.cols, x.cols);
  for (let i = 0; i < x.rows; i++) {
    for (let a = 0; a < x.cols; a++) {
      const xa = x.get(i, a);
      if (xa === 0) continue;
      for (let b = a; b < x.cols; b++)
        g.data[a * x.cols + b] = (g.data[a * x.cols + b] as number) + xa * x.get(i, b);
    }
  }
  for (let a = 0; a < x.cols; a++) for (let b = 0; b < a; b++) g.set(a, b, g.get(b, a));
  return g;
}

export function matVec(a: Matrix, v: ArrayLike<number>): Float64Array {
  if (a.cols !== v.length) throw new RangeError('matVec: shape mismatch');
  const out = new Float64Array(a.rows);
  for (let i = 0; i < a.rows; i++) {
    let s = 0;
    for (let j = 0; j < a.cols; j++) s += a.get(i, j) * (v[j] as number);
    out[i] = s;
  }
  return out;
}

export function matMul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) throw new RangeError('matMul: shape mismatch');
  const c = new Matrix(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++)
    for (let k = 0; k < a.cols; k++) {
      const aik = a.get(i, k);
      if (aik === 0) continue;
      for (let j = 0; j < b.cols; j++)
        c.data[i * b.cols + j] = (c.data[i * b.cols + j] as number) + aik * b.get(k, j);
    }
  return c;
}

export interface SymmetricEigen {
  /** Eigenvalues, largest first. */
  values: Float64Array;
  /** Eigenvectors as columns, in the same order (unit length, sign fixed so the largest entry is positive). */
  vectors: Matrix;
}

/** Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations. */
export function symmetricEigen(a: Matrix, maxSweeps = 100): SymmetricEigen {
  assertSquare(a, 'symmetricEigen');
  const n = a.rows;
  const m = a.clone();
  const v = identity(n);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += m.get(i, j) ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = m.get(p, q);
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (m.get(q, q) - m.get(p, p)) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = m.get(k, p);
          const akq = m.get(k, q);
          m.set(k, p, c * akp - s * akq);
          m.set(k, q, s * akp + c * akq);
        }
        for (let k = 0; k < n; k++) {
          const apk = m.get(p, k);
          const aqk = m.get(q, k);
          m.set(p, k, c * apk - s * aqk);
          m.set(q, k, s * apk + c * aqk);
        }
        for (let k = 0; k < n; k++) {
          const vkp = v.get(k, p);
          const vkq = v.get(k, q);
          v.set(k, p, c * vkp - s * vkq);
          v.set(k, q, s * vkp + c * vkq);
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => m.get(j, j) - m.get(i, i));
  const values = Float64Array.from(order, (i) => m.get(i, i));
  const vectors = new Matrix(n, n);
  order.forEach((src, dst) => {
    let big = 0;
    for (let k = 0; k < n; k++) if (Math.abs(v.get(k, src)) > Math.abs(v.get(big, src))) big = k;
    const flip = v.get(big, src) < 0 ? -1 : 1;
    for (let k = 0; k < n; k++) vectors.set(k, dst, flip * v.get(k, src));
  });
  return { values, vectors };
}
