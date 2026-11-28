"""Test doubles."""

from forum.budget import CostMeter
from forum.llm import LLMResponse


class ScriptedLLM:
    """Deterministic fake LLM that records every prompt it receives.

    Each role returns recognizable text (the moderator says moderator_text,
    speakers say STATEMENT-01, STATEMENT-02, ...), so tests can check exactly
    what every agent was shown.
    """

    provider = "fake"
    model_version = "fake"

    def __init__(self, moderator_text: str = "BRIEFING-TEXT", statement_text=None, stance: float = 0.6):
        self.meter = CostMeter(cap_usd=1.0)
        self.stub = True
        self.citizen_model, self.moderator_model, self.critic_model = "citizen", "moderator", "critic"
        self.calls: list[tuple[str, str, str]] = []   # (role, system, user)
        self._moderator_text = moderator_text
        self._statement_text = statement_text or (lambda n: f"STATEMENT-{n:02d} about the policy.")
        self._stance = stance
        self._n = 0

    def generate(self, model, system, user, json_mode=False, temperature=0.7, seed=None):
        if model == self.moderator_model:
            role, text = "moderator", self._moderator_text
        elif model == self.critic_model:
            role, text = "critic", '{"flagged": false, "note": ""}'
        elif json_mode:
            role = "vote"
            text = f'{{"stance": {self._stance}, "confidence": 0.7, "rationale": "fake"}}'
        else:
            self._n += 1
            role, text = "speaker", self._statement_text(self._n)
        self.calls.append((role, system, user))
        return LLMResponse(text=text, input_tokens=1, output_tokens=1)

    def prompts(self, role: str) -> list[str]:
        return [s + "\n" + u for r, s, u in self.calls if r == role]
