# ADR-001: Deliberation methodology v1 — Deliberative Polling

- **Status**: Accepted
- **Date**: 2026-05-03
- **Owners**: solo
- **Supersedes**: —

## Context

FORUM simulates structured deliberation. The methodology choice is the single most consequential design decision: it determines what the system measures, what claims it can defend, and what published precedent it can cite. The major candidates are:

1. **Deliberative Polling (DP)** — Fishkin's protocol (information presentation → small-group deliberation → opinion measurement). 30+ years of empirical and methodological work; designed specifically to measure attitude change after structured exposure to balanced information.
2. **Citizens' Assembly format (CA)** — randomly selected mini-public deliberates over multiple sessions and produces a recommendation. Politically resonant; harder to calibrate against measurable outcomes.
3. **Delphi method** — iterative anonymous expert convergence. Wrong shape for civic deliberation; participants are positioned as experts, not citizens.
4. **Structured argumentation / multi-agent debate** (Du et al., 2023) — AI-research lineage; not a peer-reviewed civic-deliberation methodology.

## Decision

v1 implements **Deliberative Polling** as the only methodology. No pluggability in v0/v1.

## Rationale

- DP is the most peer-review-defensible deliberation methodology in existence. Fishkin's published corpus (Stanford Center for Deliberative Democracy) provides citable precedent for protocol choices, sample design, and outcome-measurement claims.
- DP's measured outputs — pre/post opinion deltas with attitude-change attribution — map directly onto what FORUM emits (per-persona pre-vote and post-vote stances, plus a transcript). This makes validity claims traceable to existing literature rather than novel constructions.
- DP backtests cleanly against ballot measures: a measure has a binary outcome and a public yes/no rate, both of which DP's pre/post measurement protocol can predict.
- CA format is harder to calibrate because consensus-building is the protocol goal, not measurement of opinion change. We may add CA in v2 once DP calibration is established.
- LLM-multi-agent-debate methodologies (Du et al., 2023; Park et al., 2023, 2024) inform the *implementation*, but they are not deliberation methodologies in the political-science sense and cannot be the v1 default.

## Protocol mapping (Fishkin DP → LangGraph node graph)

The Fishkin DP protocol has four phases. Mapping to the agent graph:

| DP phase | FORUM implementation |
|---|---|
| 1. Pre-deliberation opinion measurement | `vote_node` (round 0): each citizen agent emits stance + confidence on the policy |
| 2. Information presentation (balanced briefing) | `briefing_node`: moderator agent presents a structured policy summary with the strongest arguments on each side, drawn from a curated source set |
| 3. Small-group deliberation (multiple rounds) | `deliberation_node`: rounds of speaker-selection → statement → listening; critic agent runs in parallel for fact-checking |
| 4. Post-deliberation opinion measurement | `vote_node` (round N): each citizen agent re-emits stance + confidence; delta = attitude change |

Group-size handling:
- Fishkin's protocol uses small groups (10–20) within larger deliberative samples (typically 100–500). v0 implements a single group of 12 and reports it as such. Multi-group aggregation is v1+.

## Validity considerations specific to LLM personas

DP was designed for human participants. Translating it to LLM-driven personas introduces failure modes that the methodology document must acknowledge:

1. **Briefing-balance asymmetry** — what counts as "balanced" depends on what sources the briefing draws from. The briefing source set must be documented per measure (`backtest/measures/<id>.yaml`).
2. **Speaking-time and turn-taking** — Fishkin's protocol uses moderators who balance speaking time. The FORUM moderator implements speaker rotation that approximates this.
3. **No social pressure** — real DP participants update beliefs partly under social pressure. LLM agents do not experience this directly. This is documented as a known limitation; whether it produces a systematic over- or under-update of opinions is an empirical question for the backtest.
4. **Information-fidelity asymmetry** — LLM agents have access to all training-data context, including knowledge of the actual outcome of historical measures. Backtest runs must use a measure framing that does not name the outcome year and must avoid prompting that triggers prior recall. Section 4 of `docs/methodology.md` covers the contamination protocol.

## Consequences

- v1 ships with one methodology. Customers asking for CA-format pluggability are deferred to v2.
- The backtest is the validity story; without DP-protocol fidelity, the calibration claims become indefensible.
- ADR-005 (deferred) will cover persuasion-graph methodology — currently we plan to label persuasion attribution as "self-reported" rather than implement counterfactual measurement, which is expensive. This is a known limitation.

## References

- Fishkin, J. S., & Luskin, R. C. (2005). "Experimenting with a Democratic Ideal: Deliberative Polling and Public Opinion." *Acta Politica*, 40, 284–298.
- Fishkin, J. S. (2009). *When the People Speak: Deliberative Democracy and Public Consultation*. Oxford University Press.
- Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). "Out of One, Many: Using Language Models to Simulate Human Samples." *Political Analysis*, 31(3), 337–351.
- Bisbee, J., Clinton, J., Dorff, C., Kenkel, B., & Larson, J. (2023). "Synthetic Replacements for Human Survey Data?" *Political Analysis*.
- Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). "Generative Agent Simulations of 1,000 People."
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). "Improving Factuality and Reasoning in Language Models through Multiagent Debate."
