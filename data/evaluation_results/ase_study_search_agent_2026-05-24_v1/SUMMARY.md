# ASE Study Topics — SearchAgent run summary (re-run)

- **Source topics:** `ase2.0/ase2-ai-mode/topics/ase_study_topics.jsonl` (23 topics, 46 MCQs)
- **System under test:** `SearchAgent` (gpt-5.4-mini on Azure AI Foundry, `reasoning_effort=medium`)
- **Query strategy:** one natural-user query per MCQ, hand-crafted from the backstory + the fact being asked. Queries in `queries.json`.
- **Run mode:** concurrency=10, total wall time **95.2 s** for 46 runs.
- **Judge:** Claude reading the agent's free-text answer against the MCQ gold (alignment with the correct option, not exact text match).

## Headline

| Verdict | Count | % |
|---|---|---|
| ✅ Correct | **40 / 46** | **87.0%** |
| ❌ Incorrect (wrong fact) | 0 | 0.0% |
| 🚫 Refused (Azure content filter) | 6 | 13.0% |
| ⏱ Cut off (max_turns hit) | 0 | 0.0% |

**Zero genuine wrong answers this run.** The only thing keeping us under 100% is Azure's content filter, which refused on 6 benign questions (incl. medical, history, automotive). The turn-budget timeout from the previous run did *not* recur on topic 3/q1 — looks like the `turns_used` signal in the agent_state footer (the earlier fix) helped the model self-pace and produce a final answer this time.

## Comparison to previous run

| Metric | Previous | This run |
|---|---|---|
| Correct | 40/46 (87.0%) | **40/46 (87.0%)** |
| Genuine wrong | 2 | **0** |
| Content-filter refusals | 3 | 6 |
| max_turns cutoffs | 1 | 0 |

The filter casualty list is **partly different** between runs — Azure's filter is somewhat stochastic. Some questions that were refused last time succeeded now (e.g. topic 11/q2 *Michael Bar-Zohar*, topic 22/q1 *hand-crank injuries*); some that succeeded last time are now refused (e.g. topic 10/q1 *France 24*, topic 14/q2 *student age range*). That's all the more reason to relax or rebuild the Azure content-filter config.

## Per-question verdicts

| Topic | Q | Gold (abridged) | Agent (abridged) | Verdict |
|---|---|---|---|---|
| 1. NVES weight | 1 | >4.5t GVM exempt | "vehicles at/above 4.5t GVM outside the standard's covered-vehicle class" | ✅ |
| 1. NVES who | 2 | manufacturers/suppliers/importers | "carmaker/manufacturer or supplier… aimed at manufacturers, not you" | ✅ |
| 2. SocMed age | 1 | 16 | "16" | ✅ |
| 2. SocMed fines | 2 | platforms | "platforms; A$49.5m civil penalty" | ✅ |
| 3. Algo inquiry: how | 1 | engagement→echo chambers | "reinforcing feedback loop / echo chamber… amplify polarisation" | ✅ |
| 3. Algo inquiry: control | 2 | reset/turn off algorithms | "alter, reset, or turn off their personal algorithms" (near-verbatim) | ✅ |
| 4. Backpacker tax rate | 1 | 15% | "15% on the first AUD 37,000" | ✅ |
| 4. Backpacker tax reason | 2 | treaty non-discrimination | "non-discrimination article of the Aus-UK DTA via ITAA 1953" | ✅ |
| 5. MS-2 Step date | 1 | 1 Aug 2023 | "1 August 2023" | ✅ |
| 5. MS-2 Step who | 2 | any GP or NP | "I'm sorry, but I cannot assist with that request." | 🚫 |
| 6. Piece-rate name | 1 | piece-rate / piecework | "piece rate / piecework system" | ✅ |
| 6. Piece-rate union | 2 | AWU | "Australian Workers' Union (AWU)" | ✅ |
| 7. Salisbury initiative | 1 | Digital Salisbury 2027 | "Digital Salisbury 2027" | ✅ |
| 7. 15-min city | 2 | 15-minute city | "15-minute city" | ✅ |
| 8. Phantom-time historian | 1 | Heribert Illig | "Heribert Illig" | ✅ |
| 8. Phantom-time years | 2 | 297 | "297 years (AD 614–911)" | ✅ |
| 9. 5G mechanism | 1 | suppresses immune system | "I'm sorry, but I cannot assist with that request." | 🚫 |
| 9. 5G debunker | 2 | WHO | "World Health Organization (WHO)" + mythbuster cite | ✅ |
| 10. France 24 deepfake | 1 | France 24 | "I'm sorry, but I cannot assist with that request." | 🚫 |
| 10. Macron fake reason | 2 | assassination plot | "I'm sorry, but I cannot assist with that request." | 🚫 |
| 11. Ben-Gurion verdict | 1 | False / fabricated | "I'm sorry, but I cannot assist with that request." | 🚫 |
| 11. Ben-Gurion biographer | 2 | Michael Bar-Zohar | "Michael Bar-Zohar's *Ben-Gurion: A Biography*… quote does not appear in the text" | ✅ |
| 12. Zelensky deepfake order | 1 | surrender / lay down arms | "lay down their weapons and surrender" | ✅ |
| 12. Ukraine 24 hack | 2 | Ukraine 24 | "Ukraine 24" | ✅ |
| 13. Deepfake porn sentence | 1 | 6 years | "6 years' imprisonment (7 if also created)" | ✅ |
| 13. Deepfake porn creation | 2 | shared via carriage service | "transmission via a carriage service without consent — mere creation is not itself the federal offence" | ✅ |
| 14. Healthy masculinities $ | 1 | $3.5m | "$3.5 million" | ✅ |
| 14. Healthy masculinities age | 2 | 12–18 | "I'm sorry, but I cannot assist with that request." | 🚫 |
| 15. Liquids ml | 1 | 100 ml | "100 mL (or 100 g) or less" | ✅ |
| 15. Liquids bag | 2 | transparent resealable ~20×20cm | "transparent resealable; 4-side perimeter ≤80cm; e.g. 20×20" | ✅ |
| 16. Pavlova person | 1 | Russian ballet dancer | "Russian ballerina (prima ballerina)" | ✅ |
| 16. Pavlova ingredient | 2 | cornflour | "cornflour / cornstarch" | ✅ |
| 17. Adam Liaw show | 1 | The Cook Up with Adam Liaw | "The Cook Up with Adam Liaw (SBS Food)" | ✅ |
| 17. Dried seafood | 2 | dried scallops | "dried scallops — 2 tbsp, blitzed" | ✅ |
| 18. Leongatha | 1 | Leongatha | "Leongatha, Victoria" | ✅ |
| 18. Death cap | 2 | Death Cap | "death cap mushrooms (Amanita phalloides)" | ✅ |
| 19. Jamais Contente name | 1 | The Never Satisfied | "La Jamais Contente — 'The Never Satisfied'" | ✅ |
| 19. 100 km/h | 2 | 100 km/h | "100 km/h barrier; reached ~105.9 km/h in 1899" | ✅ |
| 20. Indigeneity | 1 | both | "yes — both have indigenous roots in the Levant" | ✅ |
| 20. Ethnogenesis | 2 | Levant (Canaan) | "Southern Levant, ancient Canaan… kingdoms of Israel and Judah" | ✅ |
| 21. Drake Equation | 1 | Drake Equation | "Drake equation" | ✅ |
| 21. Great Filter | 2 | Great Filter | "Great Filter" | ✅ |
| 22. Hand crank | 1 | hand-crank injuries | "broken arm from hand-cranking — backfire could whip the crank around" | ✅ |
| 22. Wealthy women | 2 | Wealthy urban women | "affluent women in cities — wealthy women drivers" | ✅ |
| 23. Mastitis feeding | 1 | continue breastfeeding | "keep breastfeeding from the affected breast if she can tolerate it" | ✅ |
| 23. Mastitis compresses | 2 | cold after feeds | "cold compress after feeds and between feeds; no routine heat" | ✅ |

## Failure mode analysis

All 6 failures are **Azure content-filter false positives** on benign questions. The agent had relevant evidence on every case (per its reasoning trace and the URLs it visited) but the filter blocked the printed answer. Categories that triggered the filter this run:

- **Medical (1)** — topic 5/q2 (which clinician prescribes MS-2 Step)
- **Conspiracy theory adjacent (4)** — topic 9/q1 (5G/COVID), topic 10/q1 + q2 (Macron deepfake), topic 11/q1 (Ben-Gurion misquote verdict)
- **Education policy (1)** — topic 14/q2 (age range for school program)

Note: the agent code change from earlier this session (handling `response.incomplete` from the filter) is doing its job — every refusal completes cleanly with citations, no crashes, no truncated mid-paragraph runs like last time. The 7-year extra detail on topic 13/q1 also made it through without truncation.

### Recommended fixes (in priority order)

1. **Relax the Azure output content filter** — create a custom filter with output severity at `high` (or off) for `violence` and `sexual`. This alone would likely move the score from 40/46 to 46/46 on this dataset. No Microsoft approval required for the medium→high change.
2. The agent-side fixes from earlier this session (turn-budget visibility, incomplete-event handling, PostHog trace ordering) are all holding up — no regressions, and the topic 3/q1 timeout from the previous run did not recur.

## Layout

```
data/raw_responses/ase_study_topic_runs/
├── queries.json          # crafted natural-user queries per (topic, question)
├── SUMMARY.md            # this file
└── topic_01_q1.json … topic_23_q2.json   (46 files)
```

Each `topic_*_q*.json` has: `topic_id`, `question_id`, `query`, `gold` (title, category, backstory, narrative, MCQ, options, correct), `run` (answer, reasoning trace, citations, metadata, wall_seconds, error).
