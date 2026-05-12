# Fishkin & Luskin (2005) — *Experimenting with a Democratic Ideal*

**Citation**: Fishkin, J. S., & Luskin, R. C. (2005). "Experimenting with a Democratic Ideal: Deliberative Polling and Public Opinion." *Acta Politica*, 40, 284–298.

**Status**: stub from prior knowledge. Verify before citing externally.

## Core argument

Lays out the Deliberative Polling (DP) protocol and its empirical findings. DP combines a representative random sample with structured deliberation: participants complete a baseline opinion survey (T1), engage in moderated small-group deliberation with balanced briefing materials and access to expert panels, and complete a post-deliberation survey (T2). The T2-T1 delta is the "considered opinion" measurement that DP aims to estimate.

Empirical findings across many DP runs include: (a) deliberation produces systematic, replicable opinion change, often substantial; (b) the change is not driven by demographic shift in the sample, but by within-individual updating; (c) deliberation tends to increase factual knowledge; (d) policy preferences after deliberation are sometimes substantially different from initial polled opinions, and these differences tend to persist.

## Why it matters for FORUM

This paper defines the protocol FORUM v1 implements. The four-phase structure (pre-survey → briefing → small-group deliberation → post-survey) is the exact protocol mapped to the LangGraph node graph in ADR-001.

The DP literature provides:
- Validity precedent for the methodology
- Published baselines for the *magnitude* of opinion change DP produces (useful as a sanity check on FORUM's simulated deltas)
- A clear distinction between "polled opinion" (what people say when asked) and "considered opinion" (what people conclude after deliberation) — FORUM's positioning depends on this distinction

## What FORUM borrows

- Protocol structure end-to-end
- Outcome measurement format (pre/post deltas with confidence)
- The framing of DP as a measurement instrument for considered opinion, not a forecast of polling

## What FORUM does not borrow

- Real-time human moderation — FORUM uses an LLM moderator
- Expert panel Q&A — FORUM uses a curated briefing set per measure
- Multi-day in-person convening — FORUM compresses deliberation to a single agent-graph run

These departures need to be acknowledged when claiming "DP methodology" — FORUM implements a DP-shaped protocol with substantial implementation deviations.

## Open questions for the FORUM project

- How much of DP's measured opinion-change effect depends on in-person social dynamics that LLM agents cannot replicate?
- Are the magnitudes of opinion change in published DP studies a reasonable validity benchmark for simulated DP?

## Read again before

- ADR-001 sign-off
- Drafting external publication
