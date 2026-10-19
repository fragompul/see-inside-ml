import type { ParamSpec } from '@/shared/lab-kit';
import { runSurvival, type SurvivalResult, type SurvivalScenario } from './survivalCore';

export const survivalParams: ParamSpec[] = [
  {
    key: 'n',
    label: 'Subjects',
    kind: 'ladder',
    values: [50, 100, 200, 400, 800],
    format: (v) => String(v),
  },
  {
    key: 'hazardRatio',
    label: 'True hazard ratio',
    kind: 'ladder',
    values: [1, 1.5, 2.5],
    format: (v) => String(v),
    hint: 'How many times as fast the second group has events.',
  },
  {
    key: 'followup',
    label: 'Follow-up (months)',
    kind: 'ladder',
    values: [4, 8, 14, 40],
    format: (v) => String(v),
    hint: 'Everyone is watched for a random time up to this long.',
  },
];

export interface SurvivalExperiment {
  id: string;
  title: string;
  text: string;
  scenario: Partial<SurvivalScenario>;
  claim: (run: SurvivalResult) => boolean;
}

const sibling = (r: SurvivalResult, change: Partial<SurvivalScenario>) =>
  runSurvival({ ...r.scenario, ...change });
const width = (r: SurvivalResult) => (r.cox.high[0] as number) - (r.cox.low[0] as number);
const covers = (r: SurvivalResult) =>
  (r.cox.low[0] as number) <= r.trueHazardRatio && r.trueHazardRatio <= (r.cox.high[0] as number);

export const survivalExperiments: SurvivalExperiment[] = [
  {
    id: 'naive',
    title: 'Censored people are not people who had the event',
    text: 'With a follow-up of up to 14 months, 42% of the subjects are censored. The median of the observed times, which treats every censored time as an event, is 4.1 months for the first group against a true 6.9: it is too short because many of the people were still well when watching stopped. The Kaplan-Meier estimate gives 8.4 (and 2.6 for a true 2.8 in the second group).',
    scenario: {},
    claim: (r) => {
      const [a, b] = r.median;
      return (
        (a?.naive as number) < 0.65 * (a?.truth as number) &&
        Math.abs((a?.estimate as number) / (a?.truth as number) - 1) < 0.25 &&
        Math.abs((b?.estimate as number) / (b?.truth as number) - 1) < 0.25
      );
    },
  },
  {
    id: 'short-follow-up',
    title: 'A short follow-up can hide the median',
    text: 'Watched for at most 4 months, 73% are censored and the first group’s curve never falls to one half, so its median cannot be estimated at all (the true value is 6.9 months). The second group, whose events come faster, still has one (2.8 months, true 2.8).',
    scenario: { followup: 4 },
    claim: (r) =>
      (r.median[0]?.estimate as number) === Infinity &&
      Math.abs((r.median[1]?.estimate as number) - 2.8) < 0.5,
  },
  {
    id: 'log-rank',
    title: 'The log-rank test compares the whole curves',
    text: 'With 200 subjects and a true hazard ratio of 2.5 the groups differ with p well below one in a million; with a hazard ratio of 1, so that they do not, p is 0.21.',
    scenario: { n: 200, hazardRatio: 2.5 },
    claim: (r) => r.logRank.pValue < 1e-6 && sibling(r, { hazardRatio: 1 }).logRank.pValue > 0.05,
  },
  {
    id: 'hazard-ratio',
    title: 'Cox regression estimates the hazard ratio',
    text: 'The estimated hazard ratio is 3.06 against a true 2.5, with a 95% interval from 2.3 to 4.0 that contains it. Cox regression needs no assumption about the shape of the curve, only that the ratio of the two hazards stays constant.',
    scenario: { n: 400 },
    claim: (r) =>
      covers(r) && Math.abs(Math.log((r.cox.hazardRatio[0] as number) / r.trueHazardRatio)) < 0.35,
  },
  {
    id: 'sample-size',
    title: 'Few subjects, a wide interval',
    text: 'With 50 subjects the interval for the hazard ratio runs from 1.9 to 9.1 (and the estimate, 4.1, is far off); with 800 subjects it runs from 2.3 to 3.4 around 2.8. Both contain the truth; only one is useful.',
    scenario: { n: 50 },
    claim: (r) =>
      covers(r) && width(r) > 4 * width(sibling(r, { n: 800 })) && covers(sibling(r, { n: 800 })),
  },
  {
    id: 'censoring-cost',
    title: 'Heavy censoring costs information',
    text: 'Watching for at most 4 months instead of 40 turns 19% censored into 73%. The interval for the hazard ratio widens from 1.4 to 2.3 units: events that are never seen cannot be counted.',
    scenario: { followup: 4 },
    claim: (r) => width(r) > 1.5 * width(sibling(r, { followup: 40 })),
  },
];
