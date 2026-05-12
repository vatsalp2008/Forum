"""Determinism + privacy invariants of the persona schema."""

import json
import pytest

from personas.schema import DemographicSkeleton, IssuePriors, Persona, PopulationSpec


def test_persona_is_immutable():
    skel = DemographicSkeleton(
        age_band="30-44", sex="female", race_eth="white_nh",
        education="bachelors", income_band="50_75k",
        puma="11600", state="WA",
    )
    priors = IssuePriors(
        party_id="dem", p_climate_action_support=0.7,
        p_gun_restriction_support=0.6,
        p_tax_on_rich_support=0.4, ideology_score=-0.2,
    )
    p = Persona(
        persona_id="abc123",
        demographics=skel, priors=priors,
        sampling_seed=1, source_versions={"acs": "v"},
    )
    with pytest.raises(Exception):
        p.demographics.age_band = "65+"  # type: ignore[misc]


def test_population_spec_round_trips():
    spec = PopulationSpec(
        name="WA-citizens", state="WA", n=12, seed=42,
        source_versions={"acs": "5yr_2018_2022"},
    )
    d = spec.model_dump()
    s2 = PopulationSpec.model_validate(json.loads(json.dumps(d)))
    assert s2 == spec
