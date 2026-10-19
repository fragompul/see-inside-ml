import {
  createTransformer,
  type LoadedTransformer,
  type TokenizerData,
  type TransformerManifest,
} from '@sim/deep';
import manifestUrl from '@sim/deep/models/tiny-gpt/model.json?url';
import tokenizerUrl from '@sim/deep/models/tiny-gpt/tokenizer.json?url';
import weightsUrl from '@sim/deep/models/tiny-gpt/weights.bin?url';

type Fetcher = (url: string) => Promise<Response>;

async function getOk(fetcher: Fetcher, url: string): Promise<Response> {
  const response = await fetcher(url);
  if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status})`);
  return response;
}

/**
 * Downloads the tiny GPT (about 0.5 MB of weights, bundled with the app) and wires it to its
 * tokenizer. The files live in packages/deep/models/tiny-gpt and are produced by
 * tools/transformer.
 */
export async function loadTinyGpt(
  fetcher: Fetcher = (url) => fetch(url),
): Promise<LoadedTransformer> {
  const [manifest, tokenizer, weights] = await Promise.all([
    getOk(fetcher, manifestUrl).then((r) => r.json() as Promise<TransformerManifest>),
    getOk(fetcher, tokenizerUrl).then((r) => r.json() as Promise<TokenizerData>),
    getOk(fetcher, weightsUrl).then((r) => r.arrayBuffer()),
  ]);
  return createTransformer(manifest, weights, tokenizer);
}
