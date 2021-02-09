# Lifelong Commonsense Knowledge Acquisition

## Principle
+ Past commonsense knowledge should be retained and accumulated as model learns new commonsense or more sophisticated skills.

## Task Format
+ A unified language model(e.g., GPT-2) as both learner and pesudo-sample generator.
+ learning task:

    1. generative commonsense question answering.
    2. discriminative multiple-choice commonsense reasoning.
+ sample generation task:

    generate the complete commonsense questions.

## Two types of experience rehearsal
1. meta-replay by uncertain examples stored in physical episodic memory module.
2. supervised-replay by confident examples generated by the model itself.