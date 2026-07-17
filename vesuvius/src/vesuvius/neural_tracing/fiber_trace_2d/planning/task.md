# Native 3D Trace2CP Sampled CP Start Direction

Implement the todo:

- For CP start, do not use the CP tangent itself as the first trace direction.
- Sample the model direction at the CP and choose the branch/sign that most
  closely aligns with the CP-local tangent toward the target.
- Use that sampled start direction as the normal current/previous trace
  direction, so smoothness and direction supervision/scoring apply immediately
  from the first candidate step.
