WIP - ignore for now

# AGENTS.md

How to work with this project (fiber_trace_2d subproject of villa)

LLM based development process should follow this structure

# development steps

development is based around the docs in the planning/ subdir

- read (or create) task.md
- create task_plan.md
    - should plan everything described in task.md
    - must not revert on any of the specs from spec.md
    - should also follow plan.md broader plans where relevant to the current task
    - must contain a section "spec update" of what to add/remove/change to spec.md for this task
    - must contain a sections docs updates - of what to add/change in the docs/ dir
    - the plan should contain a way to testing the desired change/feature
    - plan should contain the changelog update (if relevant)
- have an agent review the task_plan.md independently against spec.md, plan.md and task.md
- create status.md
- optional depending on current working model: ask for user feedback on the plan
- implement according to plan
- run testing where relevant
- loop updates and testing until satisfied (or ask for user feedback depending on current work model)
- make sure all the relevant docs are update (docs/, status.md, task_log.md)

# some details on execution

- Deviations from the plan but also findings (successes and failures) should be logged in planning/task_log.md.
- Any simplification, partial implementation, deferred item, unsupported case,
  or intentionally skipped requirement must be reported explicitly in
  planning/task_log.md and in the final user response for that task. Silent
  simplification or silent postponement is not allowed.
- planning/task_log.md is for the current active task only. When starting a new
  task, replace its contents with that task's implementation notes, deviations,
  validation commands, and results. Do not preserve or append historical logs
  from prior tasks there; use planning/changelog.md for durable cross-task
  history when relevant.

# document details

## plan

planning/plan.md contains to overarching high level plan and some details - it is aspirational so can only partially be fulfilled

## task

planning/task.md is the current user-specified task to work on
if directly instructed to work on something this should be changed to reflect that instruction itself
if instructed to work on a item from todo.md this should be a copy of the todo item

## task_plan

planning/task_plan.md is the current detailed task-level plan for planning and implementation.

## status

planning/status.md should contain an markdown checkbox list of the development steps for the current task, including the planning steps so we can always see the current status
it should also contain relevant chunks of task_plan items at some sensible granularity

## local_development

plannning/local_development.md contains specifics about how to work with this specific checkout and system

## context

planning/context.md contains additional context information for the current task.

## changelog

planning/changelog.md should contain high level changes done grouped per date
small changes (like tweaks, parameter changes etc) do not need an entry, or iterations on a task but most tasks should probably leave at least a one line entry.

# docs

docs/ should contain code documentation at a level that should be sufficient for all high level understanding and planning as well as code details where appropriate (e.g. not all functions but important ones).
It should document the coarse code structure and where to find what functionality and how things are implemented but high level details can be found in the code itself.
