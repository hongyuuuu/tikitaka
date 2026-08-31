# Demo video — narration script and shot list

Target length **3:00**. Narration is ~450 words at a natural 150 wpm. Every
number spoken here is in `reports/m6-release-audit.json` or
`docs/baseline_results.json`; don't improvise new ones on camera.

---

## Before you hit record

- [ ] `data/catalog.jsonl` downloaded from the GitHub release (shots 4 and 5
      crash without it; shot 3 does not need it)
- [ ] Terminal font **18–20pt**, window sized so 76 columns fills the frame
- [ ] All five commands already in shell history, in order, so each shot is
      `↑` + Enter — never type on camera
- [ ] **Working tree committed** — `git status --porcelain` must print
      nothing. The release build refuses to run on a dirty tree, and a
      clean one puts `code_dirty: false` in the audit you film
- [ ] `build/` exists (`mkdir build`) — it's gitignored, throwaway
- [ ] Every command run once already, so nothing is cold
- [ ] Notifications off; Slack and Discord closed
- [ ] Recording at 1080p, mic on a separate track
- [ ] Test one 20-second take and actually listen back before committing

Commands, in the order you'll want them in history. These use `python`,
which is what this Windows machine has; the repo's other docs say `python3`
because the organizer's harness is POSIX. On Windows `python3` is a Microsoft
Store stub and will fail. `py` works too.

```bash
python scripts/replay_trace.py artifacts/traces/browsing.jsonl --pace 4
```

```bash
python -m evaluator.local_evaluator
```

```bash
python scripts/verify_m6_submission.py --archive build/submission.zip --output build/audit.json
```

---

## Shot 1 — 0:00–0:20 · Title card

> Fifty thousand products. Ten turns. One hidden target, and the customer
> starts out telling you almost nothing.
>
> This is TikiTaka, our shopping copilot for Challenge Four. I'm going to show
> you a full session, the scored run over two hundred sessions, and then prove
> the whole thing works with the language model switched off.

## Shot 2 — 0:20–0:45 · Pipeline diagram

**[On screen: the per-turn pipeline block from `ARCHITECTURE.md` §3]**

> Every turn runs the same loop. The model interprets the message into
> structured state operations. Deterministic code applies them, rebuilds the
> query from active state, and runs three retrieval routes — BM25, dense, and
> structured filters — fused with reciprocal rank fusion. Then one decision:
> ask a question, or recommend.

## Shot 3 — 0:45–1:45 · The session · **the important shot**

**[Run the replay command. Talk over the paced output.]**

> Here's a real Browsing session, replayed from the trace the agent wrote while
> it was being scored.
>
> Turn one: "Basketball Men, but I'm still exploring." Generality nine-tenths —
> that's about as vague as it gets. One hard constraint, and the agent asks
> about brand rather than guessing.
>
> Watch the mode field. Turn three, the customer names a material, and the
> agent flips itself from browsing to buying. Generality drops to nought point
> three five.
>
> It's spending these early turns deliberately. With one constraint, a
> recommendation would burn the turn — so it buys information instead, and each
> answer lands in the state as a typed constraint with the turn it came from.
>
> Turn six, four constraints in, it stops asking and commits. **Target found at
> rank one.**

## Shot 4 — 1:45–2:15 · The scored run

**[Run the evaluator. ~30s — talk through it, or cut and caption "sped up".]**

> That's one session. Here's all two hundred, on the organizer's own evaluator.
>
> Hit Rate at ten: eighty-eight and a half percent. MRR nought point five two
> nine. Mean turns to conversion, five point seven eight. Weighted technical
> score, nought point seven zero six — against a weak BM25 baseline of nought
> point one zero seven.

## Shot 5 — 2:15–2:40 · The audit

> Now the part I'd want to see if I were judging. This rebuilds the submission,
> extracts it somewhere clean, deletes the API credential, and blocks every
> socket and DNS call with a Python audit hook.
>
> Same score. Zero network attempts, zero exceptions, zero contract violations.
> Everything you just watched ran with the model switched off — the API route
> is upside, not life support.

## Shot 6 — 2:40–3:00 · Limitations and attribution

**[On screen: closing card, held for the full 20 seconds]**

> Two honest caveats. Our production dense index isn't built, so that score is
> sparse and structured retrieval only. And the medium-reasoning API default
> hasn't been measured live, so we make no cost or latency claim for it.
>
> Data is Amazon Reviews 2023 from the McAuley Lab at UCSD. Thanks for
> watching.

---

## Card text

**Opening card**

```
TikiTaka — Conversational Shopping Copilot
TikTok TechJam 2026 · Challenge 4
```

**Closing card** (all six lines on screen together)

```
Hit Rate@10  0.885     MRR  0.529     MTTC  5.78
TechnicalScore  0.706  (weak BM25 baseline: 0.107)
200 public sessions · credential removed · network blocked

Data: Amazon Reviews 2023 — McAuley Lab, UCSD
      amazon-reviews-2023.github.io
Not affiliated with or endorsed by Amazon.
```

---

## Do not film

- **`intent_override.jsonl`** — that session misses, and it shows superseded
  constraints surviving the override. Browsing (rank 1, turn 6), buying (rank
  8, turn 7), and boundary (rank 8, turn 6) all hit.
- **Any `query_summary` field** — it renders a hard constraint as
  `category!=basketball men`, which reads on camera as a negation. The replay
  script deliberately renders the structured constraint list instead.
- **Anything on amazon.com**, the Amazon logo, or the wordmark. Product titles
  from the research dataset are fine with the attribution card up.
- **Music you don't have rights to.** Silence is a perfectly good choice.

---

## Upload

1. YouTube → **Public** (not Unlisted — the rule says public)
2. Title: `TikiTaka — Conversational Shopping Copilot | TikTok TechJam 2026 Challenge 4`
3. Description: one-line summary, the Devpost URL, the repo URL, and the
   dataset attribution line from the closing card
4. Paste the watch URL into the `## Demo video` section of `DEVPOST.md`
5. Open the link in a private window to confirm it really is public
