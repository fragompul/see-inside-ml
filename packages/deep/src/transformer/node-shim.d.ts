// Tests read model files from disk. The engine tsconfig deliberately has no Node typings, so the
// few Node / platform names the tests need are declared here.
declare module 'node:fs' {
  export function readFileSync(
    path: string | URL,
  ): Uint8Array & { buffer: ArrayBuffer; byteOffset: number };
}

declare class URL {
  constructor(input: string, base?: string | URL);
}

interface ImportMeta {
  url: string;
}
