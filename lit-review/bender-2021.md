# Bender et al. (2021) — *On the Dangers of Stochastic Parrots*

**Citation**: Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?" *FAccT '21*.

**Status**: stub from prior knowledge. Verify before citing externally.

## Core argument

A foundational critique of large language models on four axes: (1) environmental and financial costs concentrate harms on populations that benefit least, (2) training corpora encode hegemonic viewpoints and amplify them, (3) the appearance of fluent, contextual language masks the absence of grounded understanding ("stochastic parrots"), (4) the technology is liable to be deployed in ways that cause real-world harm to marginalized communities.

## Why it matters for FORUM

This is the obligatory ethical-frame reference. FORUM is a politically sensitive product built on top of a class of systems that this paper documents as systemically biased. Any FORUM publication that does not engage with the Bender critique substantively will be (rightly) criticized.

The relevant arguments for FORUM:
- The critique that LLMs reproduce hegemonic viewpoints applies *directly* to FORUM's persona-rhetoric generation. The ACS demographic skeleton + ANES priors steer the *content* of opinions; the LLM still chooses the *rhetoric*, and the rhetoric is drawn from internet-scale text that over-represents specific subpopulations.
- The "stochastic parrots" framing is exactly the right framing to apply to LLM-generated deliberative arguments. A persona producing a fluent argument for a position is not the same as a person reasoning their way to that position. FORUM's outputs should be presented as transcripts of statistical-pattern-completion, not transcripts of considered reasoning.

## What FORUM borrows

- The framing discipline: every output is labeled as synthetic, statistical, and pattern-matched.
- The audit posture: known biases are documented, not hidden.
- The refusal of marketing claims that would imply more than the system can defend.

## What this paper rules out

- Any FORUM messaging that frames the system as "voice for the voiceless" or similar. The system is a *measurement instrument*, not a representation.
- Use of FORUM in contexts where the bias amplification harms documented by Bender et al. would be likely to reproduce — e.g., decisions about marginalized communities based on FORUM outputs alone.

## Open questions for the FORUM project

- How does FORUM's stratified persona sampling interact with the underlying LLM's hegemonic-viewpoint bias? Can demographic conditioning meaningfully redirect the rhetoric, or does it merely flavor a fundamentally homogenous voice?
- What audit mechanism can FORUM offer to surface bias amplification beyond per-segment error reporting?

## Read again before

- Any external publication
- AUP finalization
