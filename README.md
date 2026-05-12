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
export GEMINI_API_KEY=...               # free tier is sufficient for v0
forum personas build --state WA        # build persona library
forum deliberate i1631                 # run a single deliberation
forum backtest                          # run all measures, emit calibration report
```

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
