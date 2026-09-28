# Tune winding-step continuation on certified fine splits

Commit current state first. Use winding displacement costs on added continuation edges, not hard equality or flat break penalties. Sweep parameters and investigate formulation bugs to seek better unconditioned winding results than legacy grouped. Track iterations in task_log. Preserve certified constraints independently of splits. Approved runners only; do not increase message limits. Do not claim improvement without measurements.
