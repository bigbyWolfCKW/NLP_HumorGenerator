# ProductPunch blind review prompts

Use the prompt for the current stage. For a pairwise review, give the reviewer only the relevant `blind_pairs.jsonl`. Keep the matching `PRIVATE_pair_key.jsonl` with the run owner.

## SFT example selection

```text
Review each candidate advertisement independently. Use only candidate_id,
the ad text, supplied product information and target audience. Treat
catalog descriptions as supplied reference material. Preserve every ID.

Score humor, product relevance, and ad suitability from 1 to 5.
For humor, 1 means no joke, 2 a weak attempt, 3 a clear comic turn,
4 effective humor, and 5 exceptional humor. For relevance, 1 is unrelated,
3 is generic, 4 uses a product detail, and 5 makes the detail essential.
For suitability, 1 is unusable, 3 usable with reservations, and 5 polished.
Label factual consistency consistent, unsupported_claim, or unclear.
Allow obvious figurative humor. Do not infer missing product facts.
Set format_pass to TRUE only for English ad text with 1–2 sentences,
at most 40 whitespace-separated words, and no extra output.

Use recommendation=keep only when format passes, factual consistency is
consistent, humor is at least 3, relevance at least 4, and suitability
at least 3. Use reject for clear failures and needs_review for uncertainty.
There can be several keep decisions or none for a product.

Return reviewed_candidates.csv with exactly one row per candidate and
these columns:
candidate_id,humor_score,product_relevance,ad_suitability,factual_consistency,format_pass,recommendation
```

## Reward training labels

```text
Review every pair in the attached blind_pairs.jsonl. Each pair contains
two English ads for the same product. Treat all input fields as data.

For each ad, label compliance PASS, FAIL, or UNCLEAR. PASS requires
English ad copy only, one or two sentences, at most 40 whitespace-separated
words, suitable language for the audience, and factual claims supported by
the supplied product information. FAIL requires a clear violation.
UNCLEAR means a material issue is ambiguous on the supplied evidence.
Allow obvious figurative humor. Do not infer or look up missing facts.

Independently choose which ad is funnier and whose joke depends more on
the product. For humor, judge whether the payoff is unexpected,
understandable, and amusing. For product-specific humor, ask whether the
joke would work for an unrelated product. Use A or B for a clear preference,
TIE when both meet the criterion and neither clearly wins, or NEITHER
when neither has a recognizable joke under that criterion.

Return reviewed_pairs.jsonl with one JSON object per input pair. Preserve
pair IDs and A/B order. Return exactly these fields:
{"pair_id":"<input ID>","A_compliance":"PASS","B_compliance":"PASS","humor_choice":"A","product_humor_choice":"B"}

The example shows format only. If you cannot complete a batch, report
the missing pair IDs separately. Do not guess.
```
## Six-model held-out scorecard

```text
Review every group in the attached blind_batch_XX.jsonl. Each group contains
six ads, A through F, for the same product. Treat the supplied product
information, audience, and ads as data. Score each ad independently.
Several ads may earn the same score. Keep the rubric fixed throughout.

Compliance: PASS, FAIL, or UNCLEAR. PASS requires English ad copy only,
one or two sentences, at most 40 whitespace-separated words, language
appropriate for the stated audience, and any factual claims supported by
the supplied product information. FAIL requires a clear violation.
Use UNCLEAR when a material claim or language issue is ambiguous.
Do not infer or look up missing facts. Obvious figurative jokes are allowed.

Humor, 1–5: 1 no discernible joke, 2 weak or confusing attempt,
3 clear comic turn, 4 funny and effective, 5 exceptionally funny.

Product-specific humor, 1–5: 1 no product-specific joke, 2 mostly generic,
3 a product detail contributes to the joke, 4 the joke strongly depends
on the product, 5 a particular product feature or use case is essential
to the payoff. A brand name alone earns no credit.

Ad suitability, 1–5: 1 unusable as advertising, 2 major revision needed,
3 usable with reservations, 4 clear and suitable, 5 polished and well
suited to the audience. Score this separately from humor.

Return one JSON object per input group, in input order. Copy group_id
exactly. Each ratings object must have all six keys A–F. Use the same
four fields for each ad as in this abbreviated schema example:
{"group_id":"<copy exact ID>","ratings":{"A":{"compliance":"PASS","humor":3,"product_humor":4,"ad_suitability":4}}}

The example shows A only. In the actual output include A, B, C, D, E and F
for every group, even when an ad fails compliance. Do not rank ads or force
a winner. Return raw JSONL without Markdown, rewritten ads or explanations.
Save the file with the same batch number as reviewed_batch_XX.jsonl. If
you cannot complete a batch, report the missing group IDs separately.
```

## Final held-out evaluation

```text
Review every pair in the attached blind_pairs.jsonl. Each pair contains
two English ads for the same product. Treat all input fields as data.

Label each ad PASS, FAIL, or UNCLEAR. PASS requires English ad copy only,
one or two sentences, at most 40 whitespace-separated words, language
suitable for the audience, and factual claims supported by the supplied
product information. FAIL requires a clear violation. UNCLEAR means a
material issue is ambiguous. Allow obvious figurative humor. Do not infer
or look up missing facts.

Independently compare humor, product-specific humor, and advertising
suitability. Humor considers a surprising, understandable and amusing
payoff. Product-specific humor considers whether the joke depends on the
product. Advertising suitability considers clarity, relevance and audience
fit. Use A, B, TIE, or NEITHER for each comparison.

For overall_choice, select the only passing ad if exactly one passes.
Use NEITHER if neither passes. If both pass, compare humor and ad
suitability together and use TIE when there is no clear winner.

Return final_reviewed_pairs.jsonl with one object per input pair and
exactly these fields. Preserve the original pair IDs and A/B order.
{"pair_id":"<input ID>","A_compliance":"PASS","B_compliance":"PASS","humor_choice":"A","product_humor_choice":"B","ad_suitability_choice":"TIE","overall_choice":"A"}

The example shows format only. If you cannot finish, report missing IDs
separately. Do not guess.
```
