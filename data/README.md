# Data Card

`eduvaluebench_results.json.gz` is the analysis-ready public release used by
the offline reproduction script. It is a gzip-compressed UTF-8 JSON array with
1,696 records and 60 columns. The records contain 1,719 value assignments;
1,601 records have fine-grained scenario labels that map to the seven education
families, while 95 do not.

Each record contains:

- `id`, the stable benchmark record identifier;
- `教育场景分类点`, the fine-grained education-scenario label;
- `价值观考察点`, one or more value labels;
- `问题` and `参考回答`, the benchmark prompt and expert-authored reference
  decision with its rationale;
- one `回答_<target>` field for each of 15 target model snapshots;
- `评估分数_<target>_<judge>` fields for the common DeepSeek and Doubao judges;
- historical Hunyuan judge fields for the original ten targets only.

Scores are logged values. Valid binary decisions are numeric or string `0`/`1`;
other values are treated as evaluator non-completion by the analysis. Empty and
provider-refusal target responses are classified separately from judge
non-completion.

The release intentionally omits evaluator rationales, API logs, provider
credentials, Excel intermediates, and source worksheet names. The record IDs do
not encode annotator identities. Seven synthetic email-like strings and one
token-like string appearing in model-output fields are normalized to `[EMAIL]`
and `[REDACTED_TOKEN]`; this does not change any released score or
response-status classification. Model names identify logged API snapshots and
should not be interpreted as claims about current products.

Construction-internal challenge-mechanism tags are not stored in this
analysis-ready snapshot. Derived item scores are regenerated from the released
evaluator decisions as `outputs/latest_item_scores.csv`. The legacy output key
`abstention_rate` denotes an invalid evaluator decision or evaluator
non-completion rate. URLs appearing inside model-generated responses have not
been independently validated.
