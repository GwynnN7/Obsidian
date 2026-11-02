---
tags: [artificial-intelligence-fundamental, theory]
---

# Evolvability

The genome's capacity to **acquire and develop new features** to enhance survival and reproduction. The design goal: genomes with an inherent potential for further evolution.

## Mutation rate

- Low rates limit variation.
- High rates risk system collapse.

## Promoting it indirectly

- Change the fitness function over time.
- Use novelty search ([[Novelty and Quality-Diversity]]).
- Employ mass extinction events.

## Evolvability search (direct measurement)

- Select parents based on their **measured potential to produce diverse offspring**.
- **Test:** generate hypothetical offspring, measure the distance of their behaviour relative to the archive, then **discard** them.
- Expensive but effective; other techniques avoid evaluating the offspring by using maths instead.

> [!important] NS vs ES
> **Novelty Search** drives *individuals* towards novel behaviours.
> **Evolvability Search** drives the *offspring of individuals* towards novel behaviours.

Related: [[Novelty and Quality-Diversity]], [[Evolutionary Algorithms]].
