/** Small texts written for this lab: two to train tokenizers on and four to measure them on. */

export const PROSE_TRAIN = `The keeper of the lighthouse climbed the stairs every evening, counting the steps as his father had taught him. There were ninety-one of them, and on a stormy night each one seemed to have grown a little taller. At the top he trimmed the wick, polished the great lens and waited for the dark to settle over the water. Ships that passed in the night knew him only as a light, steady and patient, turning slowly across the waves.

In the morning the village woke to the smell of bread. The baker opened her shop before sunrise, and the first customers were always the fishermen, who bought warm loaves and talked about the weather, the tides and the price of fish. Children ran along the harbour wall on their way to school, and the old men on the benches watched them go and remembered when they had been small.

Learning to read the sea takes years. The keeper knew when a calm morning would turn to a gale, and he could tell from the colour of the sky whether the boats should stay in the harbour. He wrote everything in a thick book: the wind, the rain, the ships and the birds. Nobody ever asked to read it, but he wrote carefully all the same, because a record that nobody reads is still a record, and one day somebody might need it.`;

export const PROSE_TEST = `On the last evening of summer the keeper walked down to the harbour and bought a loaf from the baker. She asked him about the weather, and he told her that the wind was turning and that the boats would be wise to stay in port. The children were already asleep, and the village was quiet except for the sound of the water against the wall.`;

export const CODE_TRAIN = `function average(values) {
  let total = 0;
  for (let i = 0; i < values.length; i++) {
    total += values[i];
  }
  return total / values.length;
}

function maxValue(values) {
  let best = values[0];
  for (const value of values) {
    if (value > best) {
      best = value;
    }
  }
  return best;
}

const scores = [12, 7, 31, 18];
const result = { mean: average(scores), top: maxValue(scores) };
console.log(result.mean, result.top);

function filterPositive(values) {
  const out = [];
  for (const value of values) {
    if (value > 0) out.push(value);
  }
  return out;
}

export default function summary(values) {
  const positive = filterPositive(values);
  return { count: positive.length, mean: average(positive), top: maxValue(positive) };
}`;

export const CODE_TEST = `function minValue(values) {
  let best = values[0];
  for (const value of values) {
    if (value < best) {
      best = value;
    }
  }
  return best;
}

const ages = [41, 29, 35];
console.log(minValue(ages), average(ages));`;

export const SPANISH_TEST = `El farero subía las escaleras cada tarde, contando los escalones como le había enseñado su padre. Por la mañana, el pueblo despertaba con el olor del pan, y los niños corrían por el muelle camino de la escuela mientras los pescadores hablaban del tiempo y de las mareas.`;

export const NUMBERS_TEST = `Order 4815162342 shipped on 2024-11-03 for $1,299.50, arriving in 3 to 5 days. Invoice 0007319 totals 88 units at 14.75 each, or 1298.00 in all.`;

export type TrainingSet = 'prose' | 'code' | 'both';
export type TestText = 'prose' | 'code' | 'spanish' | 'numbers';

export const TRAINING: Record<TrainingSet, string[]> = {
  prose: [PROSE_TRAIN],
  code: [CODE_TRAIN],
  both: [PROSE_TRAIN, CODE_TRAIN],
};

export const TESTS: Record<TestText, { label: string; text: string }> = {
  prose: { label: 'Prose, new sentences', text: PROSE_TEST },
  code: { label: 'Code, new function', text: CODE_TEST },
  spanish: { label: 'Spanish', text: SPANISH_TEST },
  numbers: { label: 'Numbers and dates', text: NUMBERS_TEST },
};
