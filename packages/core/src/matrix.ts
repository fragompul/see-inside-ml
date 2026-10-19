/**
 * Minimal dense matrix on Float64Array (row-major).
 * Float64 on purpose: it gives bit-level parity with the NumPy reference, and the networks in the
 * playground are tiny, so precision is free and speed is a non-issue.
 */
export class Matrix {
  readonly data: Float64Array;

  constructor(
    readonly rows: number,
    readonly cols: number,
    data?: Float64Array,
  ) {
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 0 || cols < 0) {
      throw new RangeError(`Invalid matrix shape ${rows}×${cols}`);
    }
    if (data && data.length !== rows * cols) {
      throw new RangeError(`Data length ${data.length} does not match shape ${rows}×${cols}`);
    }
    this.data = data ?? new Float64Array(rows * cols);
  }

  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  static from(rows: readonly (readonly number[])[]): Matrix {
    const r = rows.length;
    const c = rows[0]?.length ?? 0;
    const m = new Matrix(r, c);
    for (let i = 0; i < r; i++) {
      const row = rows[i];
      if (!row || row.length !== c) throw new RangeError('Ragged rows are not allowed');
      m.data.set(row, i * c);
    }
    return m;
  }

  get(row: number, col: number): number {
    return this.data[row * this.cols + col] as number;
  }

  set(row: number, col: number, value: number): void {
    this.data[row * this.cols + col] = value;
  }

  clone(): Matrix {
    return new Matrix(this.rows, this.cols, this.data.slice());
  }

  toArray(): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < this.rows; i++) {
      out.push(Array.from(this.data.subarray(i * this.cols, (i + 1) * this.cols)));
    }
    return out;
  }

  /** Applies `fn` element-wise into a new matrix. */
  map(fn: (value: number) => number): Matrix {
    const out = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) out.data[i] = fn(this.data[i] as number);
    return out;
  }
}

function assertSame(a: Matrix, b: Matrix, op: string): void {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new RangeError(`${op}: shape mismatch ${a.rows}×${a.cols} vs ${b.rows}×${b.cols}`);
  }
}

/** (m×k)·(k×n) → (m×n) */
export function matmul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) throw new RangeError(`matmul: ${a.cols} ≠ ${b.rows}`);
  const { rows: m, cols: k } = a;
  const n = b.cols;
  const out = new Matrix(m, n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a.data[i * k + p] as number;
      for (let j = 0; j < n; j++) {
        out.data[i * n + j] = (out.data[i * n + j] as number) + aip * (b.data[p * n + j] as number);
      }
    }
  }
  return out;
}

/** aᵀ·b: (m×k)ᵀ·(m×n) → (k×n). Used for weight gradients. */
export function matmulAT(a: Matrix, b: Matrix): Matrix {
  if (a.rows !== b.rows) throw new RangeError(`matmulAT: ${a.rows} ≠ ${b.rows}`);
  const { rows: m, cols: k } = a;
  const n = b.cols;
  const out = new Matrix(k, n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a.data[i * k + p] as number;
      for (let j = 0; j < n; j++) {
        out.data[p * n + j] = (out.data[p * n + j] as number) + aip * (b.data[i * n + j] as number);
      }
    }
  }
  return out;
}

/** a·bᵀ: (m×k)·(n×k)ᵀ → (m×n). Used to propagate gradients to the previous layer. */
export function matmulBT(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.cols) throw new RangeError(`matmulBT: ${a.cols} ≠ ${b.cols}`);
  const { rows: m, cols: k } = a;
  const n = b.rows;
  const out = new Matrix(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++)
        sum += (a.data[i * k + p] as number) * (b.data[j * k + p] as number);
      out.data[i * n + j] = sum;
    }
  }
  return out;
}

/** Adds a length-`cols` vector to every row, in place. */
export function addRowVector(a: Matrix, v: Float64Array): void {
  if (v.length !== a.cols) throw new RangeError('addRowVector: length mismatch');
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      a.data[i * a.cols + j] = (a.data[i * a.cols + j] as number) + (v[j] as number);
    }
  }
}

/** Sum over rows → length-`cols` vector. */
export function colSums(a: Matrix): Float64Array {
  const out = new Float64Array(a.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++)
      out[j] = (out[j] as number) + (a.data[i * a.cols + j] as number);
  }
  return out;
}

export function sub(a: Matrix, b: Matrix): Matrix {
  assertSame(a, b, 'sub');
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++)
    out.data[i] = (a.data[i] as number) - (b.data[i] as number);
  return out;
}

export function mul(a: Matrix, b: Matrix): Matrix {
  assertSame(a, b, 'mul');
  const out = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++)
    out.data[i] = (a.data[i] as number) * (b.data[i] as number);
  return out;
}

export function scale(a: Matrix, factor: number): Matrix {
  return a.map((x) => x * factor);
}

/** Concatenates columns: [a | b]. */
export function hconcat(a: Matrix, b: Matrix): Matrix {
  if (a.rows !== b.rows) throw new RangeError('hconcat: row mismatch');
  const cols = a.cols + b.cols;
  const out = new Matrix(a.rows, cols);
  for (let i = 0; i < a.rows; i++) {
    out.data.set(a.data.subarray(i * a.cols, (i + 1) * a.cols), i * cols);
    out.data.set(b.data.subarray(i * b.cols, (i + 1) * b.cols), i * cols + a.cols);
  }
  return out;
}

/** Keeps columns [start, end). */
export function sliceCols(a: Matrix, start: number, end: number): Matrix {
  if (start < 0 || end > a.cols || start > end) throw new RangeError('sliceCols: out of range');
  const cols = end - start;
  const out = new Matrix(a.rows, cols);
  for (let i = 0; i < a.rows; i++) {
    out.data.set(a.data.subarray(i * a.cols + start, i * a.cols + end), i * cols);
  }
  return out;
}

/** Gathers rows by index into a new matrix. */
export function selectRows(a: Matrix, indices: ArrayLike<number>): Matrix {
  const out = new Matrix(indices.length, a.cols);
  for (let r = 0; r < indices.length; r++) {
    const src = indices[r] as number;
    if (src < 0 || src >= a.rows) throw new RangeError('selectRows: index out of range');
    out.data.set(a.data.subarray(src * a.cols, (src + 1) * a.cols), r * a.cols);
  }
  return out;
}
