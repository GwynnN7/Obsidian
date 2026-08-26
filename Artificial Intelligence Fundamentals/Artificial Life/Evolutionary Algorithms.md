---
tags: [artificial-intelligence-fundamental, alife]
---

# Evolutionary Algorithms

## The general loop

**Initialization:** a population $P_0$ of individuals with $N$ features, expressed in a domain-specific language.

**Iterative step $t$:**

1. Evaluate the fitness of the current population.
2. Select parents based on fitness.
3. Generate offspring via reproduction and feature recombination.
4. Apply mutation to the offspring features.
5. Evaluate the fitness of the new population.
6. Select the $K$ surviving individuals forming $P_{t+1}$.

**Variants:**

- **Parent survival:** replacing $K$ parents with $K$ children is common.
- **Focus:** can incorporate local search by concentrating on high-fitness regions (hill climbing - see [[Local Search]]).
- **Parameters:** different features may use different mutation rates.
- **Recombination:** averaging, taking max or min, or crossover.

## Taxonomy

| | Representation | Driven by |
|---|---|---|
| **Evolutionary Strategies (ES)** | real-valued vectors, distribution-based | **mutation**, evaluation, recombination |
| **Genetic Algorithms (GA)** | bit strings (genotype), population-based | **recombination**; mutation is infrequent; fitness drives parent selection |
| **Genetic Programming (GP)** | sets of instructions (code) | GA principles applied to code generation |

> [!note]
> These are derivative-free optimizers. They apply exactly where [[Gradient Descent]] cannot: no gradient, combinatorial or deceptive landscapes. They show up as an option for [[Hyperparameter Search]] and, via [[NEAT]], for architecture search.

Related: [[Evolutionary Strategies]], [[Genetic Algorithms]], [[Artificial Life]].
