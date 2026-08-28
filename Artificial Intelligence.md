---
tags: [moc]
---

# Artificial Intelligence

Central hub for the AI subjects. One branch per subject.

## Subjects

| Subject                                  | Notes                                    | CFU | Year |
| ---------------------------------------- | ---------------------------------------- | --- | ---- |
| **Machine Learning**                     | [[Machine Learning]]                     | 9   | 1    |
| **Artificial Intelligence Fundamentals** | [[Artificial Intelligence Fundamentals]] | 6   | 1    |
| **Computational Mathematics**            |                                          | 9   | 1    |
| **Human Language Technologies**          |                                          | 9   | 2    |
| **Computer Vision**                      |                                          | 9   | 2    |
| **Generative and Deep Learning**         |                                          | 9   | 1    |
| **Parallel and Distributed Systems**     |                                          | 9   | 1    |
| **Computational Neuroscience**           |                                          | 6   |      |
| **Continual Learning**                   |                                          | 6   |      |
| **Robotics**                             |                                          | 6   |      |
| **Algorithm Engineering**                |                                          | 9   |      |
| **Advanced Programming**                 |                                          | 9   |      |

### Quick entry

| Subject          | One page           | Canvas             |
| ---------------- | ------------------ | ------------------ |
| Machine Learning | [[ML Essentials]]  | [[ML Canvas.canvas]]  |
| AI Fundamentals  | [[AIF Essentials]] | [[AIF Map.canvas]] |

## How this vault is organized

> [!important] Folders are shelves, tags are meaning
> A note lives in **exactly one folder**, picked for browsing convenience. What a note *is* lives in its **tags**, which cut across folders and across subjects. Never create a tag that reproduces a folder name - that stores the same partition twice and tells you nothing new.

Every note carries **one subject tag** plus **one or two kind tags**. The kind vocabulary is the same in both subjects, so `#models` collects SVM and Bayesian Networks, and `#theory` collects VC dimension and entailment.

| Tag | What it marks |
|---|---|
| `#machine-learning` / `#artificial-intelligence-fundamental` | the subject |
| `#models` | a representation you can instantiate - it defines a hypothesis space or a structure |
| `#algorithms` | a procedure with steps |
| `#theory` | concepts, definitions, properties, bounds, proofs - nothing runnable |
| `#training` | how the free parameters are found, and the knobs that control it |
| `#validation` | measuring, selecting, estimating |
| `#reference` | lookup material: notation, algebra, guideline tables |
| `#moc` | an index note - a hub, a folder map, an Essentials page |

A note can hold two kind tags when it genuinely is both: [[Backpropagation]] is `#training` and `#algorithms`, [[NEAT]] is `#algorithms` and `#models`.

> [!note]
> Because the tags carry the meaning, a note does **not** have to move when its kind differs from its shelf. [[Universal Approximation]] is a theorem that lives in the Neural Networks folder, and that is fine - it is tagged `#theory` and the search finds it.

## Cross-subject threads

Things that show up in more than one subject - link them here as they appear.

| Thread                             | Machine Learning                             | AI Fundamentals                                                      |
| ---------------------------------- | -------------------------------------------- | -------------------------------------------------------------------- |
| **Local optima, restarts, noise**  | [[Gradient Descent]], [[Multiple Minima]]    | [[Local Search]] - hill climbing, simulated annealing                |
| **Derivative-free search**         | [[Hyperparameter Search]], [[Random Search]] | [[Evolutionary Algorithms]], [[Evolutionary Strategies]]             |
| **Architecture search**            | [[Cascade Correlation]]                      | [[NEAT]], HyperNEAT                                                  |
| **Bayes rule**                     | [[Bayes Optimal Classifier]]                 | [[Probability and Decision Theory]], [[Bayesian Networks]]           |
| **State summarizing the past**     | [[RNN]]                                      | [[Hidden Markov Models]], [[Reasoning over Time]]                    |
| **Convolution / weight sharing**   | [[CNN]]                                      | [[Cellular Automata]] (NCA)                                          |
| **Hypothesis space in logic**      | [[Concept Learning]], [[Version Space]]      | [[Propositional Logic]], [[First-Order Logic]]                       |
| **No bias, no generalization**     | [[Inductive Bias]]                           | [[Agent Architectures]] - the table agent                            |
| **Structure buys tractability**    | -                                            | [[Tree-Structured CSPs]], [[Bayesian Network Inference]] (polytrees) |
| **The i.i.d. assumption breaking** | [[Generalization and Risk]]                  | [[Reasoning over Time]] - concept drift                              |
| **Agents that learn**              | [[Reinforcement Learning]]                   | [[Agent Architectures]], [[Monte-Carlo Tree Search]]                 |
| **Causality is not correlation**   | [[ML Workflow]]                              | [[Causal Networks]] - the do-operator                                |
| **Symbolic vs distributed codes**  | [[Distributed Representation]]               | [[Propositional Logic]], [[First-Order Logic]] - symbols share nothing |
| **Graphs as model vs as data**     | [[Structured Domain Learning]], [[Deep Graph Networks]] | [[Bayesian Networks]], [[Causal Networks]]                |
| **Discretizing a continuous space**| [[Vector Quantization]], [[SOM]]             | [[Novelty and Quality-Diversity]] - the MAP-Elites bins              |

