"""
LLM-as-judge for the explanation-trust annotation set.

The LLM receives exactly what a human annotator sees (patient record, model
prediction and confidence, LIME explanation) and the same two questions, and
never sees the true label or whether the prediction was correct.

    pip install openai pandas
    set OPENAI_API_KEY=sk-...            (Windows PowerShell: $env:OPENAI_API_KEY="sk-...")
    python src/llm_judge.py --model gpt-4o-mini --repeats 3

Writes annotation/llm_judgements.csv (one row per item per repeat).
--dry-run prints the prompt for the first item without calling the API.
"""
import argparse
import json
import os
import time

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'annotation')

SYSTEM = (
    "You are taking part in a study on explanations of machine-learning predictions. "
    "A model estimates whether a patient has heart disease from 13 clinical attributes. "
    "For each prediction you are shown the patient record, the model's prediction and confidence, "
    "and an automatic explanation (LIME) listing up to 5 facts about the patient and whether each "
    "pushed the model towards DISEASE or NO DISEASE; a larger weight means a stronger push. "
    "Answer two questions exactly as a careful human annotator would:\n"
    "Q1 'How much do you trust this prediction?' on a 1-5 scale "
    "(1 = strongly distrust, 2 = distrust, 3 = unsure, 4 = trust, 5 = strongly trust).\n"
    "Q2 'Does the explanation make sense given the patient record?' Yes or No.\n"
    "Respond with JSON only: {\"trust\": <integer 1-5>, \"makes_sense\": \"Yes\" or \"No\", "
    "\"reason\": \"<one sentence>\"}"
)


def user_prompt(row):
    return (f"Patient record:\n{row.patient}\n\n"
            f"Model prediction: {row.prediction} (confidence {row.confidence:.0%})\n\n"
            f"Explanation (LIME, top 5 facts):\n{row.explanation}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--repeats', type=int, default=3,
                    help='independent calls per item, to measure the judge\'s self-consistency')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    items = pd.read_csv(os.path.join(ROOT, 'items.csv'))
    if a.dry_run:
        print(SYSTEM, '\n---\n', user_prompt(items.iloc[0]))
        return
    from openai import OpenAI
    client = OpenAI()
    out_path = os.path.join(ROOT, 'llm_judgements.csv')
    rows = []
    for rep in range(1, a.repeats + 1):
        for row in items.itertuples():
            for attempt in range(5):
                try:
                    r = client.chat.completions.create(
                        model=a.model, temperature=a.temperature,
                        response_format={'type': 'json_object'},
                        messages=[{'role': 'system', 'content': SYSTEM},
                                  {'role': 'user', 'content': user_prompt(row)}])
                    j = json.loads(r.choices[0].message.content)
                    trust = int(j['trust']); assert 1 <= trust <= 5
                    sense = 'Yes' if str(j['makes_sense']).strip().lower().startswith('y') else 'No'
                    break
                except Exception as e:                         # rate limits / malformed JSON
                    print(f'  retry {row.item_id} ({e})'); time.sleep(2 * (attempt + 1))
            else:
                raise SystemExit(f'failed on {row.item_id}')
            rows.append(dict(item_id=row.item_id, repeat=rep, model=a.model, trust=trust,
                             makes_sense=sense, reason=j.get('reason', '')))
            print(rep, row.item_id, trust, sense)
        pd.DataFrame(rows).to_csv(out_path, index=False)   # save after each pass
    print(f'wrote {len(rows)} judgements to {out_path}')


if __name__ == '__main__':
    main()
