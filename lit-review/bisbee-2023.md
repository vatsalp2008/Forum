# Bisbee et al. (2023) — *Synthetic Replacements for Human Survey Data?*

**Citation**: Bisbee, J., Clinton, J., Dorff, C., Kenkel, B., & Larson, J. (2023). "Synthetic Replacements for Human Survey Data? The Perils of Large Language Models." *Political Analysis*.

**Status**: stub from prior knowledge. Verify before citing externally.

## Core argument

Push-back on Argyle (2023). Bisbee et al. show that LLM "synthetic respondents" can produce systematically biased estimates compared to real survey data — specifically, that the algorithmic-fidelity claim is sensitive to (a) the prompt phrasing, (b) the model version, and (c) the demographic subpopulation. Some subpopulations are simulated well; others are simulated badly, with the bias direction stable enough to produce misleading aggregate results.

## Why it matters for FORUM

This paper is the obligatory critical companion to Argyle. Any FORUM publication that cites Argyle without also citing Bisbee will be dismissed by reviewers. The Bisbee findings argue specifically for FORUM's per-segment error reporting — aggregate calibration can hide subpopulation-level bias, and the methodology document already commits to per-segment breakdowns precisely for this reason.

## What FORUM borrows

- The claim that calibration must be reported *per demographic segment*, not just aggregate.
- The framing of LLM-survey-simulation as a "tool with documented biases" rather than a substitute for real data.
- The methodological discipline of stating limitations alongside claims.

## What this paper rules out

- Marketing claims of the form "FORUM accurately represents what voters think." Bisbee's evidence is sufficient to make such claims indefensible.
- Single-aggregate-number reports without per-segment error.

## Open questions for the FORUM project

- Are Bisbee's bias findings on *one-shot survey simulation* relevant to FORUM's *deliberation* setting? Plausibly yes (same model, same demographic conditioning), but the deliberation setting may amplify or dampen the biases. Empirical question.
- Which demographic segments are likely to be the worst-calibrated in FORUM, by analogy to Bisbee's findings?

## Read again before

- ADR-002 sign-off
- Backtest report drafting
- Any external publication
