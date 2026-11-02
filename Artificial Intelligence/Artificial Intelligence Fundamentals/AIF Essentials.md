---
tags: [artificial-intelligence-fundamental, moc]
---

# AIF Essentials

The whole subject on one page. One question runs through all of it: **what should the agent do next?** Each stage is a different answer.

The visual version is [[AIF Map.canvas]]. The folder-by-folder view is [[Artificial Intelligence Fundamentals]].

## The five lines

1. An agent perceives and acts; **rational** means acting to maximize expected performance given what it knows.
2. If you can enumerate states, **search** for the action sequence.
3. If the state has internal structure, **exploit the structure** - it is what turns exponential into tractable.
4. If someone else is also choosing, search **their** choices too.
5. If you cannot enumerate, **represent** instead - in logic when you know things for sure, in probability when you do not.

## The path

### 1. What an agent is

$\to$ [[Rational Agents]], [[Environments]], [[Agent Architectures]]

**Keep:** rationality is about the **expected** outcome given the percepts, not about being right. And the six environment dichotomies decide which technique from the rest of this page is even applicable.

### 2. Deciding by searching

$\to$ [[Search Problems]], [[Tree Search]], [[Uninformed Search]]

**Keep:** every algorithm here is the same loop with a different **frontier discipline**; judge them on completeness, optimality, time and space.

### 3. Using an estimate of what is left

$\to$ [[Informed Search]], [[Local Search]]

**Keep:** A\* is optimal iff the heuristic is **admissible** (never overestimates); for graph search you also need **consistency**. Local search gives up completeness to get memory back.

### 4. Structured problems

$\to$ [[Constraint Satisfaction Problems]], [[Backtracking Search]], [[Constraint Propagation]], [[Tree-Structured CSPs]]

**Keep:** the state stops being a black box. Inference (propagation) and ordering heuristics do far more work than raw search - and a **tree-shaped** constraint graph is solvable in linear time.

### 5. When someone else is choosing too

$\to$ [[Adversarial Search]], [[Minimax]], [[Alpha-Beta Pruning]], [[Heuristic Game Search]], [[Monte-Carlo Tree Search]], [[Stochastic Games]]

**Keep:** alpha-beta returns **exactly** the minimax value, it only skips nodes that cannot change it. When you have no decent evaluation function, sample instead: MCTS.

### 6. Many agents, no single objective

$\to$ [[Multiagent Systems]], [[Game Theory]], [[Social Welfare]]

**Keep:** an equilibrium is a *stability* concept, not a *goodness* one - Nash and Pareto are different questions, and prisoner's dilemma is the standing counterexample.

### 7. Search without gradients

$\to$ [[Evolutionary Algorithms]], [[Genetic Algorithms]], [[Evolutionary Strategies]], [[NEAT]], [[Novelty and Quality-Diversity]], [[Evolvability]], [[Cellular Automata]]

**Keep:** population-based search only needs to *evaluate*, never to differentiate. Its failure mode is a deceptive objective, and the fix is to stop optimizing the objective directly ([[Novelty and Quality-Diversity]]).

### 8. Knowing instead of searching

$\to$ [[Knowledge-Based Agents]], [[Entailment and Models]], [[Logical Inference]], [[Propositional Logic]], [[First-Order Logic]], [[FOL Inference]]

**Keep:** separate **syntax** (what you can write) from **semantics** (what it means) from **inference** (what you can derive). Sound = never derives something false; complete = derives everything entailed.

### 9. Acting on what you know

$\to$ [[Classical Planning]], [[PDDL]], [[Planning Heuristics]], [[Advanced Planning]], [[Planning in Non-Deterministic Environments]]

**Keep:** planning is search where the *representation of the state is open*, so the heuristic can be derived automatically by **relaxing the problem**. Non-determinism turns a plan from a sequence into a conditional tree.

### 10. Acting under uncertainty

$\to$ [[Probability and Decision Theory]], [[Bayesian Networks]], [[Bayesian Network Inference]], [[Causal Networks]]

**Keep:** a Bayesian network is compact only because of **conditional independence** - that is the whole trick. Inference is exact on polytrees and approximate (sampling) otherwise. And correlation is not causation: interventions need the do-operator.

### 11. Uncertainty over time

$\to$ [[Reasoning over Time]], [[Hidden Markov Models]]

**Keep:** the Markov assumption is what makes the state finite; then filtering, smoothing, prediction and most-likely-sequence are four different questions on the same model.

## The rules you don't break

> [!important] Admissibility
> A\* with an inadmissible heuristic is just fast, not optimal. Consistency is the stronger condition you need for graph search.

> [!important] The structure is the point
> Tractability comes from structure, not from cleverness: tree-shaped CSPs, polytree Bayesian networks, independence assumptions. Same idea three times.

> [!important] Representation before inference
> Decide what the state, the language and the independence assumptions are. Every algorithm on this page is downstream of that choice.

## Where this meets Machine Learning

The shared threads are listed in [[Artificial Intelligence]] - local optima and restarts, derivative-free search, architecture search, Bayes rule, state summarizing the past, symbolic vs distributed representations, and graphs as model vs as data.
