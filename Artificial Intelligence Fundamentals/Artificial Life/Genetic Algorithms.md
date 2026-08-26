---
tags: [artificial-intelligence-fundamental, alife]
---

# Genetic Algorithms

Candidate solutions are represented by a **chromosome string** (the genotype), from which the **phenotype** is generated.

Effective for optimizing high-dimensional combinatorial functions, or landscapes with multiple local optima.

## Input encoding

**Incorrect encoding can preclude solutions or bias the search** - it is the design decision that matters most.

- **Gray encoding:** only a single bit differs between consecutive integers, fixing the Hamming-distance problem of plain binary (where 7 $\to$ 8 flips every bit).
- **Direct encoding:** each genotype element maps directly to a phenotype feature. Easy to design, scales poorly - the search space has the same dimension as the phenotype.
- **Indirect encoding:** codes the **developmental rule** for the phenotype. Complex to design, but gives modularity and flexibility, and searches a space **smaller** than the phenotype space.

## Operators

- **Selection:** parents chosen proportionally to fitness.
- **Crossover (recombination):** 1-point (a single split point) or 2-point (two split points, switching the intermediate segment).
- **Mutation:** infrequent, random modification of a single element (a bit flip).

Related: [[Evolutionary Algorithms]], [[NEAT]], [[Data and Preprocessing]] (encoding matters there too).
