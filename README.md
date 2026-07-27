# FORUM

**A research artifact, not a product (yet).**

FORUM simulates structured deliberation among demographically grounded synthetic personas, using deliberative-polling methodology, and reports the result with full transparency about its synthetic nature. The v0 goal is one defensible artifact: a backtest of FORUM's predictions against past Washington State ballot measures, with full methodology, code, and per-measure error tables.

## What FORUM is

A multi-agent LLM system that:
- Samples synthetic personas from joint distributions of public demographic microdata (ACS PUMS) and conditions issue-position priors on published survey microdata (ANES).
- Runs a deliberative-polling protocol (Fishkin) over those personas: information presentation → small-group deliberation → opinion measurement.
- Produces a deliberation transcript, a weighted opinion-change report, and a per-segment outcome breakdown.
- Backtests its predictions against resolved ballot measures and reports calibration error.

## What FORUM is not

- **Not a prediction of what real people think.** Personas are statistical composites; outputs are simulated deliberative outcomes, not survey results.
- **Not a representation of any individual.** No persona corresponds to any real person.
- **Not an election forecaster.** FORUM does not predict named live elections or referenda.
- **Not a messaging tool.** FORUM refuses to generate persuasive messaging targeted at any demographic.
- **Not production-grade software.** v0 is a CLI-driven research artifact.

See [docs/methodology.md](docs/methodology.md) for the full methodological position.

## Status

v0, in development. Solo-maintained. Free-tier-only stack.

## Quick start

```bash
# Requires Python 3.11+
uv sync                                 # install dependencies
echo "GEMINI_API_KEY=..." > .env        # default model: gemini-2.5-flash-lite
forum personas build --state WA        # build persona library (needs ACS+ANES data)
forum deliberate i1631                 # run a single deliberation
forum backtest                          # run all measures, emit per-measure reports
forum sensitivity --n 12 --seeds 1,2,3,4,5
                                        # sensitivity sweep across seeds (with CIs)
forum contamination-probe              # measure LLM prior-knowledge leakage
forum refusal-check "predict the next election"
                                        # test the input-side refusal layer
```

**Population scope.** Personas are sampled per-state by default; load ACS for
multiple states and sample nationally (population-weighted) with
`forum personas sample --state US`. The v0 measure set is WA-only, so national
sampling matters once national measures are added.

A single deliberation in stub mode (no API key needed) is the fastest way to
verify everything works: `forum deliberate i1631 --stub`. The full pipeline
runs end-to-end and produces a report with pseudo-random stances. Every report
is stamped `Mode: STUB` or `Mode: LIVE` so stub output is never mistaken for a
real result.

**Live runs and quota.** Live mode calls the model API and fails loudly on
errors rather than fabricating data — set `stub=True` for offline runs. Google's
free tier caps requests per day (as low as ~20/day for some models), which is
below a single full deliberation; a real backtest sweep needs a billed key.
Tune pacing with `FORUM_LLM_MIN_INTERVAL_S` and `FORUM_LLM_MAX_RETRIES`.

**Cross-model robustness (v2).** Runs can use a second model family for the
cross-model audit in [docs/methodology.md](docs/methodology.md) §5.6:

```bash
echo "ANTHROPIC_API_KEY=..." >> .env
forum deliberate i1631 --provider anthropic   # default claude-opus-5
FORUM_ANTHROPIC_MODEL=claude-haiku-4-5 forum deliberate i1631 --provider anthropic
```

The Anthropic path meters real token usage, so the `--budget` cap actually
binds (unlike the free Gemma tier). A Claude API key needs credit balance —
the free "Evaluation access" plan has none.

## Repository layout

```
docs/                     methodology, ADRs
  adr/                    architecture decision records
lit-review/               summaries of foundational sources
personas/                 schema, sampler, priors loaders
forum/                    agent graph (LangGraph DP protocol)
backtest/                 harness, metrics, measure ground-truth
  measures/wa/            per-measure YAML definitions
tests/                    pytest suite
```

## Acceptable use

FORUM is licensed under Apache 2.0, but the system itself refuses certain uses. See `docs/aup.md`. In short:

- No election prediction within 60 days of a named vote
- No persuasive-messaging generation
- No targeting of named demographics for conversion
- Public-sector deployments must publish full deliberation logs within 90 days

## Contributing / advising

This is an open methodological project. If you are a political scientist, public-policy methodologist, or civic-tech researcher and want to advise, open an issue or email the maintainer.

## License

Apache 2.0 — see [LICENSE](LICENSE).
