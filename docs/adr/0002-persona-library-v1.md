# ADR-002: Persona library v1 — data sources, sampling, and re-identification posture

- **Status**: Accepted
- **Date**: 2026-05-03
- **Owners**: solo

## Context

Personas in FORUM are not LLM-invented stereotypes. They are stratified samples drawn from real demographic microdata, with issue-position priors derived from public survey microdata. The choice of source data determines:

- **Legal posture** — which sources permit derived works at all
- **Re-identification risk** — small-cell sampling on microdata can in principle re-identify individuals
- **Methodological defensibility** — peer reviewers will ask about the source distribution
- **Geographic coverage** — v0 is WA-only

## Decision

v0 uses **two** sources only:

1. **ACS PUMS 5-year 2018–2022** — for the joint demographic distribution (skeleton)
2. **ANES 2020 Time Series Study** — for issue-position priors

GSS, CES, and Pew ATP are deferred to v1+.

State voter files, commercial polling proprietary data, and any social-platform-scraped data are excluded categorically.

## Source analysis

### ACS PUMS 5-year 2018–2022 (Census Bureau)

- **License**: Public domain. U.S. government work, 17 U.S.C. § 105.
- **Use**: Joint demographic distribution at PUMA (Public Use Microdata Area) level. Variables sampled: AGEP, SEX, RAC1P, HISP, SCHL, PINCP, HINCP, TEN (housing tenure), PUMA, ST.
- **Citation**: U.S. Census Bureau, American Community Survey Public Use Microdata Sample, 5-year 2018–2022. Acquired from data.census.gov.
- **Re-identification posture**: Census already applies disclosure-avoidance noise to PUMS. We additionally enforce **k-anonymity ≥ 5** on every persona sample: any combination of demographic attributes used to draw a persona must correspond to at least 5 records in the source data. The sampler refuses to emit personas from cells smaller than k=5.

### ANES 2020 Time Series Study

- **License**: Free for research use. The ANES data agreement permits use and aggregation but **restricts redistribution of the microdata**. We may distribute *derived* aggregates (cross-tab priors at sufficient cell sizes) but not the raw microdata. Citations are required in any work using ANES.
- **Use**: Issue-position priors. Variables include party ID, ideology, issue-stance items relevant to climate, firearms, education, and tax policy (the v0 backtest measure topics).
- **Citation**: American National Election Studies. 2021. *ANES 2020 Time Series Study Full Release* [dataset and documentation]. anes.org. The persona library stores per-cell prior tables only; no raw ANES record is shipped or queryable.
- **Acquisition**: Requires registration at electionstudies.org. The user (researcher) is responsible for downloading and placing the data file at `personas/data/raw/anes_2020_timeseries.dta`. The repo does not include ANES microdata.

## Sampling protocol

Algorithm: **stratified sampling with k-anonymity enforcement.**

1. Define the population (e.g., "Washington State adults, citizen, 2018–2022 ACS basis"). The population definition is a YAML in `personas/populations/`.
2. Define stratification cells over (age band × sex × race/ethnicity × education × income band × PUMA region). Cells smaller than k=5 are merged with adjacent cells until k≥5 or refused.
3. Draw N personas proportional to cell weight, with deterministic seed.
4. For each demographic skeleton, look up issue-position priors from the ANES cross-tab table conditioned on (party ID × education × age band) — coarsened to maintain ANES cell sizes.
5. The full persona record is: `{demographic_skeleton, issue_priors, sampling_seed, source_versions, persona_id}`.

The sampler is deterministic given (seed, population spec, source versions). Reproducibility is a hard requirement.

## Re-identification posture (formal)

FORUM commits in writing that:

- No persona corresponds to any real individual.
- Persona demographic skeletons are sampled from cells with k≥5 in the source data.
- Persona records are released only as part of a deliberation report, never as standalone identity profiles.
- The system refuses to sample at cell sizes below k=5 even if explicitly instructed to do so.

This is encoded as an assertion in `personas/sample.py`. Tests verify the assertion fires.

## Excluded sources (with reasoning)

- **State voter files** — varies by state. WA permits some access but with restrictions on derived products and on use that could constitute "voter targeting." Per-state legal review required; not on the v0 critical path.
- **Commercial polling proprietary data** — license terms typically prohibit derived products.
- **Social-platform-scraped data** — TOS violations, biased toward online subpopulations, ethically questionable.
- **GSS, CES, Pew ATP** — usable, but ANES is sufficient for the v0 measure topics. Adding more sources increases v0 surface area without changing the core validity question.

## Consequences

- v0 personas are WA-residents-conditioned only. National personas are v2.
- The k=5 enforcement reduces the diversity of personas the system can emit at small N. This is a deliberate trade-off favoring privacy guarantees over persona granularity.
- Adding GSS / CES / Pew in v1 is straightforward; the sampler architecture supports new prior tables without code changes to the agent graph.

## References

- U.S. Census Bureau. *American Community Survey Public Use Microdata Sample, 2018–2022 5-year*. https://www.census.gov/programs-surveys/acs/microdata.html
- American National Election Studies. *2020 Time Series Study*. https://electionstudies.org
- Sweeney, L. (2002). "k-anonymity: A model for protecting privacy." *International Journal on Uncertainty, Fuzziness and Knowledge-Based Systems*, 10(5).
