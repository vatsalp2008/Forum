# FORUM methodology

This document is the public-facing methodology statement. It describes what FORUM does, what it does not do, what claims it makes, and what claims it refuses.

## 1. What FORUM is

FORUM is a multi-agent LLM system that simulates a Deliberative Polling protocol (Fishkin & Luskin, 2005; Fishkin, 2009) on a population of synthetic personas. The personas are stratified samples drawn from joint distributions of public demographic microdata (American Community Survey PUMS) with issue-position priors derived from the American National Election Studies (ANES). The output is a transcript of structured deliberation, pre/post opinion measurements per persona, and an aggregate report with confidence intervals and per-segment breakdowns.

## 2. What FORUM is not

- **FORUM does not represent real people.** Every persona is a statistical composite drawn from cells with at least k=5 records in the source microdata. No persona corresponds to any identifiable individual.
- **FORUM does not predict survey results.** It simulates *deliberative* outcomes, which differ from polled opinions because deliberation changes minds (this is the substantive claim of the DP literature).
- **FORUM does not predict named live elections or referenda within 60 days of the vote.** The system refuses such requests on the output side.
- **FORUM does not generate persuasive messaging.** It refuses requests to optimize messaging for any demographic and refuses framings that frame personas as targets for conversion.

## 3. The protocol

For a given policy question:

1. **Population specification** — the user names a population (e.g. "Washington State citizen adults, ACS 2018–2022 basis"). This determines which demographic distribution the personas are drawn from.
2. **Persona sampling** — the sampler draws N personas (default 12 for v0) proportional to the cell weights in the population's joint distribution, with k≥5 anonymity enforcement.
3. **Prior conditioning** — each persona's issue-position prior is looked up from the ANES cross-tab table, conditioned on a coarsened set of demographic axes (party ID × education × age band).
4. **Pre-deliberation vote** — each persona emits its initial stance on the policy question with confidence.
5. **Briefing** — the moderator agent presents a balanced briefing on the policy. The briefing's source set is documented per measure and reviewable.
6. **Deliberation rounds** — for R rounds (default R=5), the moderator selects speakers, each speaker emits a statement, all personas listen and may update internal beliefs. A critic agent runs in parallel, fact-checking each statement against a grounded source set.
7. **Post-deliberation vote** — each persona re-emits its stance with confidence.
8. **Report synthesis** — pre/post deltas, weighted aggregate, per-demographic-segment breakdown, confidence intervals.

## 4. Reproducibility and contamination control

- **Determinism**: every deliberation specifies a random seed, model version, prompt version, persona library version, and source-data versions. Re-running with identical specs produces identical output (modulo any model-side non-determinism, which is recorded).
- **Backtest contamination protocol**: when running against historical ballot measures, the briefing must not include the actual outcome. Measure YAMLs in `backtest/measures/wa/` define the framing the moderator may use, scrubbed of outcome information. We additionally instruct the moderator to refuse acknowledgment of historical outcomes during deliberation. **This control is imperfect** — LLMs trained on web text have substantial prior knowledge of well-known historical measures. The backtest report flags this as a known confound.
- **Contamination probe** (`forum contamination-probe`): to *quantify* the leakage above rather than merely warn about it, we probe the model directly. Out of character and with no briefing or deliberation, the model is asked whether it already knows the measure's certified outcome and, if so, the certified yes-percentage. We compute `recall_error = |model_recalled_yes_pct − actual_yes_pct|` and flag each measure HIGH / MODERATE / LOW / NONE. A backtest prediction that matches the real result on a HIGH-contamination measure should be discounted as potential memorization rather than treated as an emergent deliberative result. The probe is run per (measure, seed) and its report lives alongside the run artifacts.

## 5. Validity claims and limitations

### What FORUM claims

- A *predicted yes-rate* with a stated confidence interval for any well-formed policy question on the v0 backtest measure set.
- A *per-demographic-segment* breakdown of predicted opinion change.
- *Calibration error* (Brier score, MAE) on the backtest measure set.
- A *sensitivity range* across persona seeds (sample-to-sample variance).

### What FORUM does not claim

- That its predictions match what real voters would do under real deliberation.
- That its predictions match polling.
- Any claim about a specific named future election or measure.
- Any claim about an identifiable individual or sub-N=5 demographic segment.

### Known limitations

1. **LLM-priors contamination**: LLM agents have substantial training-data knowledge of historical measures. The backtest is a *partial* validity test, not a clean prediction.
2. **Online-discourse bias**: even with demographic skeletons and ANES priors, the LLM-generated rhetorical content reflects online discourse patterns more than offline civic discourse.
3. **No social pressure**: real DP participants update beliefs partly under interpersonal social pressure. LLM agents do not experience this directly.
4. **Briefing balance is model-dependent**: what counts as a "balanced" briefing depends on the moderator agent's behavior. We document the briefing source set per measure and report any flagged imbalances.
5. **Persuasion-graph attribution is self-reported, not causal**: FORUM does not yet implement counterfactual measurement of statement-level persuasion. Reported "influence" is the agent's self-attribution of belief change, which is rationalization-prone.
6. **Single model family**: v0 uses Gemini only. Cross-model robustness audit is v2.

## 6. Refusal of service — outputs that FORUM never produces

FORUM refuses, on the output side regardless of input framing:

- Probability-of-passage forecasts on named live ballot measures or candidates within 60 days of vote.
- Messaging recommendations or persuasion strategies targeted at any demographic.
- Single-demographic deliberations whose framing is "how do we convert group X."
- Predictions about named individuals.
- Any output framed as a substitute for, rather than a supplement to, real public consultation.

These refusals are encoded in the Composer Agent (input-side check) and Output Synthesizer (output-side check). The Output Synthesizer's refusal is the binding one.

## 7. Citations

- Fishkin, J. S., & Luskin, R. C. (2005). Experimenting with a Democratic Ideal: Deliberative Polling and Public Opinion. *Acta Politica*, 40, 284–298.
- Fishkin, J. S. (2009). *When the People Speak*. Oxford University Press.
- Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of One, Many: Using Language Models to Simulate Human Samples. *Political Analysis*, 31(3), 337–351.
- Bisbee, J., Clinton, J., Dorff, C., Kenkel, B., & Larson, J. (2023). Synthetic Replacements for Human Survey Data? *Political Analysis*.
- Park, J. S., et al. (2024). Generative Agent Simulations of 1,000 People.
- U.S. Census Bureau. American Community Survey Public Use Microdata Sample, 2018–2022.
- American National Election Studies. 2020 Time Series Study.
