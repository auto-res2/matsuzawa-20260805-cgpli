You are an independent auditor extracting parameter values from an academic paper, for later
cross-checking against a separately generated reproduction. This is a standalone task: base every
value only on what the paper itself states in `REPRO_DIR/paper.txt` (and, if present,
`REPRO_DIR/tex_src/` — prefer it for exact table numbers and macro definitions).

Task:
- Read the paper and find the value it states for each parameter name listed under
  `PARAMETER_KEYS` at the end of this prompt.
- Write `REPRO_DIR/paper_extraction.json`:
  ```json
  {"parameters": [{"name": "learning_rate", "value": "0.01"}]}
  ```
- Record a number as digits only (no units, %, commas, or abbreviations like "1.3k"/"1e-6").
  Record a name (model/dataset/method) as its plain form (e.g. "CIFAR-10", "Adam") without
  parenthetical qualifiers.
- Include only the parameters the paper actually states; omit any key it doesn't mention (do not
  guess or infer a plausible value).

Tool Use:
- Available tools: Read, Grep, Glob, Write.

Output:
- Make the file change directly in the workspace. Do not ask for permission; proceed autonomously.

REPRO_DIR:
PARAMETER_KEYS:
