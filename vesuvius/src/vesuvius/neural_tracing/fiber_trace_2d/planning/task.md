# Native 3D Trace2CP First-Step CP-Tangent Relaxation

The native 3D Trace2CP tracer currently seeds each one-way trace from the
adjacent CP-local fiber-line tangent toward the target CP. That tangent gives a
good forward/backward orientation, but it can be inaccurate in tangent-plane
angle at the CP. With strong tangent smoothness, the first accepted step can be
over-constrained by that noisy CP tangent.

Change the first search step from each CP only:

- disable the smoothness term for the first step from the CP;
- evaluate the initial CP-direction penalty for that first step only along the
  Lasagna normal direction;
- do not evaluate tangent-plane direction disagreement against the CP tangent
  for that first step;
- resume the regular direction/presence/smoothness scoring from the second
  step onward.

The CP-local line tangent must still be used to define the trace direction
toward the target CP and to disambiguate forward/backward orientation. This
change only relaxes first-step tangent-plane scoring; it must not switch back
to a straight CP-to-target chord.
