---
tags: [artificial-intelligence-fundamental, logic]
---

# Knowledge-Based Agents

KBAs maintain an internal **representation of the world**, use **inference** to reason (generating new facts and selecting actions), and store domain-specific knowledge in a **Knowledge Base (KB)**.

## Knowledge representation

- **Declarative:** encodes *what* the agent knows.
- **Procedural:** encodes *how* the agent acts.
- The agent interacts with the KB through **Tell** (add knowledge) and **Ask** (retrieve information).

## Formal language

Each sentence in the KB is expressed in a formal language defined by its:

- **Syntax:** the structure of well-formed sentences.
- **Semantics:** the precise meaning, i.e. a truth value, of each sentence.

Related: [[Entailment and Models]], [[Propositional Logic]], [[Agent Architectures]].
