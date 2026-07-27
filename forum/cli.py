"""FORUM command-line interface.

Commands:
    forum personas build --state WA      # ingest ACS+ANES into the persona DB
    forum personas info                  # show loaded sources
    forum deliberate <measure_id>        # run a single deliberation
    forum backtest [--measure ...]       # run backtest on resolved measures
    forum refusal-check "<framing>"      # test the refusal layer

All commands accept --stub to run without LLM calls (uses canned outputs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table


def _load_dotenv() -> None:
    """Tiny .env loader. No dependency on python-dotenv."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

from backtest.measure_loader import list_measures
from backtest.run import run_all, run_multigroup, run_one, run_sensitivity
from forum.refusal import RefusalError, check_request
from personas.db import connect, get_source_versions

app = typer.Typer(help="FORUM — synthetic deliberative-polling research artifact.")
personas_app = typer.Typer(help="Persona library management.")
app.add_typer(personas_app, name="personas")

console = Console()


@personas_app.command("build")
def personas_build(
    state: str = typer.Option("WA", help="Two-letter state code"),
    skip_acs: bool = typer.Option(False, help="Skip ACS load (priors only)"),
    skip_anes: bool = typer.Option(False, help="Skip ANES load (skeleton only)"),
    cces: bool = typer.Option(False, help="Also load CCES 2018 priors (blended with ANES)"),
) -> None:
    """Load ACS PUMS and ANES (optionally CCES) into the persona library."""
    con = connect()
    if not skip_acs:
        from personas.acs_loader import load_state
        n = load_state(con, state=state)
        console.print(f"[green]ACS:[/green] loaded {n} cells for {state}")
    if not skip_anes:
        from personas.anes_loader import load_anes
        np_, pp = load_anes(con)
        console.print(f"[green]ANES:[/green] loaded {np_} prior cells, {pp} party-id cells")
    if cces:
        from personas.cces_loader import load_cces
        nc = load_cces(con)
        console.print(f"[green]CCES:[/green] loaded {nc} prior cells (blended with ANES)")


@personas_app.command("info")
def personas_info() -> None:
    """Show loaded persona-library source versions."""
    con = connect()
    versions = get_source_versions(con)
    if not versions:
        console.print("[yellow]No sources loaded. Run `forum personas build`.[/yellow]")
        return
    t = Table("source", "version")
    for k, v in versions.items():
        t.add_row(k, v)
    console.print(t)


@personas_app.command("sample")
def personas_sample(
    state: str = typer.Option("US", help="Two-letter state, or 'US' for a national sample"),
    n: int = typer.Option(12, help="Number of personas"),
    seed: int = typer.Option(42),
) -> None:
    """Preview a population sample (state or national) as a demographic summary."""
    from collections import Counter

    from personas.sample import sample_personas
    from personas.schema import PopulationSpec

    con = connect()
    versions = get_source_versions(con)
    if not versions:
        console.print("[yellow]No sources loaded. Run `forum personas build`.[/yellow]")
        raise typer.Exit(code=1)
    spec = PopulationSpec(
        name=f"{state}-adult-citizens", state=state, n=n, seed=seed,
        source_versions=versions,
    )
    personas = sample_personas(con, spec)
    by_state = Counter(p.demographics.state for p in personas)
    by_party = Counter(p.priors.party_id for p in personas)
    console.print(f"[green]{len(personas)}[/green] personas ({'national' if state.upper() in ('US', '*') else state})")
    console.print(f"  states: {dict(by_state)}")
    console.print(f"  party:  {dict(by_party)}")


@app.command()
def deliberate(
    measure_id: str = typer.Argument(..., help="Measure id, e.g. wa_i1631 or i1631"),
    n: int = typer.Option(12, help="Number of personas"),
    seed: int = typer.Option(42),
    stub: bool = typer.Option(False, help="Run without LLM calls (canned outputs)"),
    budget: float = typer.Option(5.0, help="USD budget cap for this run"),
    provider: str = typer.Option("gemini", help="Model family: gemini | anthropic"),
) -> None:
    """Run a single deliberation on a measure."""
    measure_id = measure_id if measure_id.startswith("wa_") else f"wa_{measure_id}"
    try:
        report = run_one(measure_id, n_personas=n, seed=seed, stub=stub,
                         budget_usd=budget, provider=provider)
    except RefusalError as e:
        console.print(f"[red]REFUSED:[/red] {e}")
        raise typer.Exit(code=2)
    console.print(report.render())


@app.command()
def backtest(
    measures: list[str] = typer.Option(None, help="Restrict to specific measure ids"),
    n: int = typer.Option(12),
    seed: int = typer.Option(42),
    stub: bool = typer.Option(False, help="Run without LLM calls"),
    provider: str = typer.Option("gemini", help="Model family: gemini | anthropic"),
) -> None:
    """Run backtest on all (or some) resolved measures."""
    mids = None
    if measures:
        mids = [m if m.startswith("wa_") else f"wa_{m}" for m in measures]
    try:
        reports = run_all(measure_ids=mids, n_personas=n, seed=seed, stub=stub, provider=provider)
    except RefusalError as e:
        console.print(f"[red]REFUSED:[/red] {e}")
        raise typer.Exit(code=2)
    console.print(f"[green]Wrote {len(reports)} reports.[/green]")


@app.command()
def sensitivity(
    measures: list[str] = typer.Option(None, help="Restrict to specific measure ids"),
    n: int = typer.Option(12),
    seeds: str = typer.Option("1,2,3,4,5", help="Comma-separated seed list"),
    stub: bool = typer.Option(False, help="Run without LLM calls"),
) -> None:
    """Run a sensitivity sweep: each measure x each seed, with aggregate stats."""
    mids = None
    if measures:
        mids = [m if m.startswith("wa_") else f"wa_{m}" for m in measures]
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    rows = run_sensitivity(measure_ids=mids, n_personas=n, seeds=seed_list, stub=stub)
    for r in rows:
        spread = r.max_predicted - r.min_predicted
        console.print(
            f"[green]{r.measure_id}[/green] | actual {r.actual_yes_pct:.1f}% "
            f"| mean predicted {r.mean_predicted:.1f}% (sd {r.stdev_predicted:.2f}) "
            f"| spread {spread:.1f} pts | mean MAE {r.mean_mae:.2f}"
        )


@app.command()
def multigroup(
    measure_id: str = typer.Argument(..., help="Measure id, e.g. wa_i1631 or i1631"),
    groups: int = typer.Option(5, help="Number of independent deliberating groups"),
    size: int = typer.Option(12, help="Personas per group"),
    seed: int = typer.Option(42, help="Base seed (group g uses seed+g)"),
    stub: bool = typer.Option(False, help="Run without LLM calls"),
    budget: float = typer.Option(15.0, help="USD budget cap for this run"),
) -> None:
    """Run several independent groups on a measure and aggregate them."""
    measure_id = measure_id if measure_id.startswith("wa_") else f"wa_{measure_id}"
    try:
        report = run_multigroup(
            measure_id, n_groups=groups, group_size=size, seed=seed,
            stub=stub, budget_usd=budget,
        )
    except RefusalError as e:
        console.print(f"[red]REFUSED:[/red] {e}")
        raise typer.Exit(code=2)
    console.print(report.render())


@app.command("persuasion-graph")
def persuasion_graph(
    measure_id: str = typer.Argument(..., help="Measure id, e.g. wa_i1631 or i1631"),
    n: int = typer.Option(12, help="Number of personas"),
    seed: int = typer.Option(42),
    stub: bool = typer.Option(False, help="Run without LLM calls"),
    provider: str = typer.Option("gemini", help="Model family: gemini | anthropic"),
) -> None:
    """Counterfactual leave-one-speaker-out influence on the final vote (§5.5)."""
    from backtest.persuasion import run_persuasion_graph

    measure_id = measure_id if measure_id.startswith("wa_") else f"wa_{measure_id}"
    try:
        rows = run_persuasion_graph(measure_id, n_personas=n, seed=seed,
                                    stub=stub, provider=provider)
    except RefusalError as e:
        console.print(f"[red]REFUSED:[/red] {e}")
        raise typer.Exit(code=2)
    for r in rows:
        console.print(
            f"[green]{r.speaker_id}[/green] | {r.n_statements} stmts "
            f"| |shift| {r.mean_abs_shift:.3f} | signed {r.mean_signed_shift:+.3f}"
        )


@app.command("contamination-probe")
def contamination_probe(
    measures: list[str] = typer.Option(None, help="Restrict to specific measure ids"),
    seeds: str = typer.Option("1", help="Comma-separated seed list"),
    stub: bool = typer.Option(False, help="Run without LLM calls"),
) -> None:
    """Probe the model's cold prior knowledge of each measure's outcome.

    Measures LLM-priors contamination — the backtest's central confound.
    """
    from backtest.contamination import run_contamination_probe

    mids = None
    if measures:
        mids = [m if m.startswith("wa_") else f"wa_{m}" for m in measures]
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    results = run_contamination_probe(measure_ids=mids, seeds=seed_list, stub=stub)
    for r in results:
        my = f"{r.model_yes_pct:.1f}%" if r.model_yes_pct is not None else "—"
        color = "red" if r.flag in ("HIGH", "MODERATE") else "green"
        console.print(
            f"[{color}]{r.measure_id}[/{color}] (seed {r.seed}) | knows={r.model_knows} "
            f"| model {my} vs actual {r.actual_yes_pct:.1f}% | flag {r.flag}"
        )


@app.command("list-measures")
def cmd_list_measures() -> None:
    """List available measure YAML files."""
    for m in list_measures():
        console.print(f"- {m}")


@app.command("refusal-check")
def refusal_check(framing: str = typer.Argument(..., help="Framing to test")) -> None:
    """Test the input-side refusal check on a framing string."""
    res = check_request(framing)
    if res.refused:
        console.print(f"[red]REFUSED:[/red] {res.reason}")
        raise typer.Exit(code=1)
    console.print("[green]OK:[/green] framing passes the input-side refusal check.")


@app.command()
def info() -> None:
    """Show repo layout / status."""
    docs = [
        "README.md",
        "docs/methodology.md",
        "docs/aup.md",
        "docs/adr/0001-methodology-deliberative-polling.md",
        "docs/adr/0002-persona-library-v1.md",
    ]
    for d in docs:
        marker = "[green]ok[/green]" if Path(d).exists() else "[red]missing[/red]"
        console.print(f"- {d}  {marker}")


if __name__ == "__main__":
    app()
