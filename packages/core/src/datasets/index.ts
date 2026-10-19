export type { ClassicDataset } from './types';
export { iris } from './iris';
export { wine } from './wine';
export { diabetes } from './diabetes';
export { breastCancer } from './breastCancer';

import { breastCancer } from './breastCancer';
import { diabetes } from './diabetes';
import { iris } from './iris';
import { wine } from './wine';

export const CLASSIC_DATASETS = { iris, wine, diabetes, breastCancer } as const;
export type ClassicDatasetId = keyof typeof CLASSIC_DATASETS;
