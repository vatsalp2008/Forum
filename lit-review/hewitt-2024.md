# Hewitt et al. (2024) — *Predicting Results of Social Science Experiments Using LLMs*

**Citation**: Hewitt, L., Ashokkumar, A., Ghezae, I., & Willer, R. (2024). "Predicting Results of Social Science Experiments Using Large Language Models."

**Status**: stub from prior knowledge. Verify before citing externally.

## Core argument

The authors test whether GPT-4 can predict the outcomes of real, published social-science experiments — given the experimental setup but blinded to the outcome — and evaluate prediction accuracy across many studies. The empirical finding is that LLM-generated predictions correlate non-trivially with real experimental outcomes, often outperforming naive baselines, though with substantial residual error and systematic biases on certain study types.

## Why it matters for FORUM

Most directly relevant to the calibration claim FORUM wants to make. Hewitt et al. provide:
- A precedent for the *form* of validity claim FORUM should target ("LLM-driven predictions correlate with real outcomes at level X, with residual MAE Y, with biases on dimensions Z")
- A precedent for the *level of claim* one can defensibly make: not "LLMs predict outcomes" but "LLM predictions are useful as a noisy signal alongside other evidence"
- A reference for thinking about prompt engineering of the prediction task: how the question is framed materially affects accuracy

## What FORUM borrows

- The disclosure-standard template ("predictions correlate at r = ..., MAE = ..., systematic biases on ...")
- The framing of LLM predictions as decision-support, not authoritative truth
- The honesty about residual biases

## What FORUM does not assume

- Hewitt et al. are predicting *experimental results* (often within-subject manipulations on online samples). FORUM is predicting *deliberative outcomes* — different shape, possibly different difficulty.
- Hewitt et al. use single-shot prediction. FORUM uses multi-agent multi-round simulation, which has its own failure modes.

## Open questions for the FORUM project

- What baseline are FORUM's predictions compared against in the backtest? Naive baselines: (a) 50%, (b) prior-poll average, (c) ANES-prior aggregated without deliberation. The backtest harness should report all four for context.
- Are FORUM's biases similar in shape to those Hewitt et al. found, or different in a way that suggests deliberation simulation has different failure modes than experiment-result prediction?

## Read again before

- Backtest report drafting
- Any external publication
