# Task: separate prepass and winding Defect controls

Split the orientation-prepass Mixed unary cost from the later winding-stage
Defect unary cost. `--bp-mixed-cost` must configure only the initial H/V/Mixed
BP, while a separate `--winding-defect-cost` configures Defect during winding.

When fixed-orientation winding is used, save the exact H/V/Defect assignment
passed from the orientation prepass into the winding solver as separate OBJ
layers, in addition to the existing final winding-state OBJ layers.
