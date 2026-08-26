---
tags: [artificial-intelligence-fundamental, moc]
---

# Artificial Life

Simulating real biological processes with computational or physical mechanisms. The process is iterative and driven by **evolutionary principles**.

**Classifications:** *Soft* ALife (digital simulation), *Hard* ALife (robotics), *Wet* ALife (artificial cells).

## The notes

- [[Evolutionary Algorithms]] - the general loop, and the taxonomy.
- [[Evolutionary Strategies]] - black-box optimization, CMA-ES.
- [[Genetic Algorithms]] - encoding, crossover, mutation.
- [[NEAT]] - evolving network topologies, CPPN, HyperNEAT.
- [[Novelty and Quality-Diversity]] - when the objective is deceptive.
- [[Evolvability]] - evolving the capacity to evolve.
- [[Cellular Automata]] - complex behaviour from local rules.

## Digital evolution

Transition from an **initial condition** to an output via an evolutionary algorithm. Essential components:

- **Differential reproduction:** survival determined by relative fitness - survival of the "fit **enough**".
- **Heredity:** traits transmitted to successive generations.
- **Variation:** mechanisms ensuring diversity.

**Dynamics:**

- **Punctuated equilibria:** evolution can proceed rapidly, alternating with periods of stasis.
- **Adaptation:** a feature favoured by natural selection becomes progressively more significant.
- **Exaptation:** a trait developed for one purpose is co-opted for a different use.

**Sources of diversity (the divergent process)** - ensures novelty and prevents population collapse: random mutation, recombination (crossover, genetic shuffling), natural selection driven by the genotype's efficacy in gene transmission (fitness).

Part of [[Artificial Intelligence Fundamentals]].
