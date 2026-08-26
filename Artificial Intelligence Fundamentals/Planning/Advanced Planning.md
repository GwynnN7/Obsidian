---
tags: [artificial-intelligence-fundamental, planning]
---

# Advanced Planning

## Hierarchical planning (HSP)

Mitigates the **action explosion** problem by abstracting away details.

- **HLA (High Level Action):** admits one or more **refinements** into a sequence of actions; its implementation uses primitive actions.
- **HLP (High Level Plan):** its implementation concatenates the implementations of all its HLAs. The HLP achieves the goal if **at least one** of its implementations succeeds.
- **Hierarchical forward planning:** start with an HLP containing a single HLA called `Act`, then use BFS to find possible refinements for each HLA in the current plan. Results are typically human-understandable.

## Sensorless planning

- Maintain a **belief state**: the set of possible states the agent might be in.
- **Open-world assumption:** if a fluent does not appear, its value is **unknown** - the opposite of the closed-world assumption of [[PDDL]].
- **Conditional effects** - `when condition: effect`:
  - `condition` is a logical formula compared against the current belief,
  - `effect` describes the resulting state,
  - an action's effect may depend on the unknown state; all the conditional effects whose `when` clause is verified get applied.

See also [[Planning in Non-Deterministic Environments]].

## Online planning (planning as you go)

Requires a new plan through **execution monitoring**. Needed when there are too many contingencies to prepare for, when an unprepared contingency occurs, or when the agent's model of the world proves incorrect.

| Monitoring | Checks before execution |
|---|---|
| **Action monitoring** | all the preconditions hold |
| **Plan monitoring** | the remaining plan will succeed |
| **Goal monitoring** | whether a better set of goals exists |

## Time, schedules and resources

Classical planning ignores time - both the duration of an action and its timing.

- Actions can have a **duration** and constraints, and may need a specific order.
- **Resource constraints:** resources defined by type, number and consumable/reusable status. **Aggregation** represents them as numerical quantities, which reduces complexity.
- **Planning phase** selects actions and ordering constraints; the **scheduling phase** adds the temporal information so the plan meets its deadlines.

### Partial-Order Plan (POP) and critical path

A POP is a path in a directed graph allowing **parallel** actions.

- The **critical path** is the longest-duration path: it determines the total plan duration.
- Actions off the critical path have **slack**: a window defined by an **Earliest Start (ES)** and a **Latest Start (LS)**, which together constitute a schedule for the problem.
- A greedy **minimum-slack heuristic** works well, scheduling for the earliest possible start at each iteration.

Related: [[Classical Planning]], [[Multiagent Systems]], [[Reasoning over Time]].
