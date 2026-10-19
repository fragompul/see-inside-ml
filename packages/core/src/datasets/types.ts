export interface ClassicDataset {
  id: string;
  title: string;
  licence: string;
  kind: 'classification' | 'regression';
  features: string[];
  classes: string[] | null;
  rows: number[][];
  target: number[];
}
