# Argyle et al. (2023) — *Out of One, Many*

**Citation**: Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). "Out of One, Many: Using Language Models to Simulate Human Samples." *Political Analysis*, 31(3), 337–351.

**Status**: stub from prior knowledge. Verify before citing externally.

## Core argument

GPT-3, when conditioned on demographic backstories drawn from real survey respondents (ANES), produces survey-response distributions that closely match the real survey-response distributions of those demographics. The authors call this property "algorithmic fidelity": a language model can be steered to imitate human subpopulations whose joint demographic distribution it is conditioned on.

## Why it matters for FORUM

This is the most directly relevant precedent for FORUM's design. The persona-library architecture (ACS demographic skeleton + ANES issue-position priors → LLM-conditioned persona prompt) is essentially a Fishkin-DP-shaped extension of Argyle's setup. If algorithmic fidelity holds for ANES-style survey items, the v0 backtest should produce non-trivial predictive signal.

## What FORUM borrows

- The pattern of conditioning the LLM on a demographic backstory drawn from real survey respondents.
- The validity claim format: "the simulated population's response distribution matches the real distribution within X% on N items."

## What FORUM should not assume

- Argyle's results are on *survey items*. FORUM is predicting *post-deliberation* outcomes — a different and harder target.
- Argyle's prompt format is single-shot Q&A. FORUM uses multi-round structured deliberation, which introduces drift and turn-taking dynamics not tested in the original paper.
- Algorithmic fidelity has been challenged subsequently — see Bisbee et al. (2023).

## Open questions for the FORUM project

- Does algorithmic fidelity transfer from survey items to ballot-measure questions framed in deliberative context?
- How does fidelity degrade as you move from a one-shot survey response to a 5-round deliberation?

## Read again before

- Drafting any external publication or grant application
- ADR-002 final sign-off on persona-library architecture
