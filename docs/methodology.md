# FORUM methodology

This document is the public-facing methodology statement. It describes what FORUM does, what it does not do, what claims it makes, and what claims it refuses.

## 1. What FORUM is

FORUM is a multi-agent LLM system that simulates a Deliberative Polling protocol (Fishkin & Luskin, 2005; Fishkin, 2009) on a population of synthetic personas. The personas are stratified samples drawn from joint distributions of public demographic microdata (American Community Survey PUMS), restricted to adult citizens and, for backtests, weighted toward likely voters using validated turnout (CCES 2018). Party identification comes from the American National Election Studies (ANES), calibrated to each state's partisan balance (CCES). Issue-position priors blend ANES and CCES. The output is a transcript of structured deliberation, pre/post opinion measurements per persona, and an aggregate report with confidence intervals, baselines, and per-segment breakdowns.

## 2. What FORUM is not

- **FORUM does not represent real people.** Every persona is a statistical composite drawn from cells with at least k=5 records in the source microdata. No persona corresponds to any identifiable individual.
- **FORUM does not predict survey results.** It simulates *deliberative* outcomes, which differ from polled opinions because deliberation changes minds (this is the substantive claim of the DP literature).
- **FORUM does not predict named live elections or referenda within 60 days of the vote.** The system refuses such requests on the output side.
- **FORUM does not generate persuasive messaging.** It refuses requests to optimize messaging for any demographic and refuses framings that frame personas as targets for conversion.

## 3. The protocol

For a given policy question:

1. **Population specification** — the user names a population (e.g. "Washington State likely voters, ACS 2018–2022 basis"). Non-citizens (ACS `CIT` = 5) are excluded by default, since ballot measures are decided by citizens. Backtests weight the pool toward *likely voters*: each demographic cell's weight is multiplied by its validated turnout propensity, P(voted | age × education × race), estimated from the CCES 2018 vote validated against voter files (not self-report).
2. **Persona sampling** — the sampler draws N personas (default 12 for v0) proportional to the cell weights, with k≥5 anonymity enforcement by *merging*: a cell with fewer than 5 source records is pooled with the other thin cells of the same demographic combination into a statewide cell (location generalized), and only combinations still under 5 statewide are refused. Dropping thin cells instead would not be demographically neutral, because rare combinations are disproportionately minority residents (it removed 77% of Black Washingtonians). With merging, 99.9% of the population is sampleable and every race group appears at its true share.
3. **Prior conditioning** — party ID is drawn from the ANES conditional P(party | age × education × race), then raked per state so the state's adult-citizen pool matches that state's party balance (CCES 2018, survey-weighted). The national conditional alone made Washington personas more Republican than Democratic. Issue-position priors are then looked up from the ANES + CCES cross-tabs (party ID × education × age band), blended by cell size.
4. **Pre-deliberation vote** — each persona emits its initial stance on the policy question with confidence, before seeing any information. This doubles as the "no deliberation" baseline.
5. **Briefing** — the moderator agent presents a balanced briefing, generated from the measure's curated neutral facts, pro/con arguments, and source list. From then on the briefing is pinned into every citizen's context, so later statements never push it out of view.
6. **Deliberation rounds** — for R rounds (default R=5), the moderator selects speakers. Each speaker sees the pinned briefing plus the six most recent citizen statements and emits a statement. A critic agent fact-checks each statement against the curated briefing facts.
7. **Post-deliberation vote** — each persona re-emits its stance with confidence, seeing the pinned briefing and the full transcript of the discussion.
8. **Report synthesis** — pre/post deltas, weighted aggregate, per-demographic-segment breakdown, confidence intervals, and baselines (see §5).

## 4. Reproducibility and contamination control

- **Determinism**: every deliberation specifies a random seed, model version, prompt version, persona library version, and source-data versions. Everything on FORUM's side is reproducible across runs and machines: persona draws, speaker order, every per-agent seed (derived with a SHA-256 hash, never Python's per-process salted `hash()`), and stub output. A test reruns the same seed under different interpreter hash salts and requires identical votes. Live model sampling is not reproducible, because the provider APIs as used do not accept a seed. Live runs therefore report variance across seeds (`forum sensitivity`) rather than relying on a single run.
- **Backtest contamination protocol**: when running against historical ballot measures, no agent may see the outcome *or the measure's identity*. Measure YAMLs in `backtest/measures/wa/` define a framing scrubbed of outcome information. Beyond that:
  - The measure's title is never sent to any agent.
  - Its ballot number ("Initiative 1631", "I-1631") and every year are stripped from all curated material: framing, briefing facts, source citations, and pro/con arguments.
  - Model output (the moderator's briefing and citizen statements) is scrubbed of ballot identifiers and of the measure's own number and year before any other agent sees it.
  - The moderator is told to treat the policy as a proposal under consideration, without mentioning results, polling, ballot numbers or dates.

  A test records every prompt of a full deliberation for each measure and fails if the ballot number, year, or title appears. **This control is still imperfect**: a model can recognize a well-known measure from its substance alone, which the contamination probe below measures.
- **Contamination probe** (`forum contamination-probe`): to *quantify* the leakage above rather than merely warn about it, we probe the model directly. Out of character and with no briefing or deliberation, the model is asked whether it already knows the measure's certified outcome and, if so, the certified yes-percentage. We compute `recall_error = |model_recalled_yes_pct − actual_yes_pct|` and flag each measure HIGH / MODERATE / LOW / NONE. A backtest prediction that matches the real result on a HIGH-contamination measure should be discounted as potential memorization rather than treated as an emergent deliberative result. The probe is run per (measure, seed) and its report lives alongside the run artifacts.

## 5. Validity claims and limitations

### What FORUM claims

- A *predicted yes-rate* with a stated confidence interval for any well-formed policy question on the v0 backtest measure set.
- A *per-demographic-segment* breakdown of predicted opinion change.
- *Calibration error* (Brier score, MAE) on the backtest measure set.
- A *sensitivity range* across persona seeds (sample-to-sample variance).
- *Baselines* for every prediction: the same personas' pre-deliberation vote (does deliberation add accuracy?), the survey prior alone with no LLM (where a survey item matches the topic: climate, firearms), and a 50% coin flip. A prediction that does not beat these carries no evidence of skill.

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
5. **Persuasion-graph attribution**: `forum persuasion-graph` measures speaker-level influence *counterfactually* — it re-runs every persona's final vote with one speaker's statements removed and attributes the resulting stance shift to that speaker (measured, not self-reported). The counterfactual context differs from the real one *only* by that speaker's lines; the briefing is pinned in both. A placebo re-vote (nothing removed, different seed) sets a noise floor, and only speakers above it show a measurable effect. Limits: one placebo draw per persona; a leave-one-out on the final vote, not a full re-deliberation; speaker-level rather than statement-level.
6. **Single model family**: v0 uses Gemini only. The harness now also supports the Anthropic/Claude family (`--provider anthropic`), so a run's model family is recorded in every report's model version (e.g. `anthropic:claude-opus-5`) and results can be compared across families. The full cross-model robustness audit — running the backtest set on both families and reporting the divergence — is v2 and requires a billed key on each provider.
7. **Electorate model**: likely-voter weights come from 2018 (midterm) validated turnout and are applied to every measure, including the presidential-year 2012 and 2016 measures, whose electorates were larger and somewhat younger. Turnout propensities are national (age × education × race), not state-specific. Party calibration uses one survey year (CCES 2018) for all measures. Alaska has too few CCES respondents for a party target and keeps the national distribution.
8. **Small rooms are noisy**: a single 12-persona room can draw an unrepresentative mix by chance (for example 7 of 12 Republicans in a state where the calibrated rate is about 35%). Use multi-group runs (`forum multigroup`) or several seeds before reading anything into one room.

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
- Ansolabehere, S., & Schaffner, B. F. (2019). CCES Common Content, 2018. Harvard Dataverse. doi:10.7910/DVN/ZSBZ7K
