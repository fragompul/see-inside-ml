import { type BpeTokenizer } from '@sim/deep';
import { cn } from '@/design-system';
import { tokenHue, tokenLabel } from '../tokens';

interface Props {
  ids: readonly number[];
  tokenizer: BpeTokenizer;
  /** Tokens before this index came from the prompt; the rest were generated. */
  promptLength: number;
  selected: number | null;
  onSelect: (position: number | null) => void;
  /** Hide the token ids under the chips (compact mode). */
  showIds?: boolean;
  label?: string;
}

/** The text as the model sees it: one coloured chip per token, with its id underneath. */
export function TokenStrip({
  ids,
  tokenizer,
  promptLength,
  selected,
  onSelect,
  showIds = true,
  label = 'Tokens',
}: Props) {
  return (
    <ul className="flex flex-wrap gap-x-1.5 gap-y-2.5" aria-label={label}>
      {ids.map((id, position) => {
        const generated = position >= promptLength;
        const active = selected === position;
        return (
          <li key={position} className="flex flex-col items-center gap-0.5">
            <button
              type="button"
              aria-pressed={active}
              aria-label={`Token ${position + 1}: ${tokenizer.tokenString(id)} (id ${id})`}
              onClick={() => onSelect(active ? null : position)}
              style={{ ['--h' as string]: tokenHue(position) }}
              className={cn(
                'tok cursor-pointer rounded-md px-1.5 py-0.5 font-mono text-[0.88rem] leading-snug whitespace-pre transition-all duration-200',
                'hover:-translate-y-0.5',
                active && 'tok-solid text-bg shadow-[0_0_0_3px_var(--brand-soft)]',
                generated && !active && 'border-dashed',
              )}
            >
              {tokenLabel(tokenizer.tokenString(id), id)}
            </button>
            {showIds && (
              <span className="font-mono text-[0.6rem] leading-none text-faint tabular-nums">
                {id}
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}
