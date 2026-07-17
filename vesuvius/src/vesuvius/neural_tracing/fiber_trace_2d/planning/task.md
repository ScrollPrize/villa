# Native 3D Trace2CP Cumulative Tangent Smoothness

Add a cumulative smoothness penalty to native 3D Trace2CP so several small
tangent-plane turns cannot combine into a large unwanted turn. The penalty is
smoothness-only and tangent-only:

- keep the existing local smoothness term;
- keep the normal/elevation behavior unchanged;
- add a short-history tangent-plane direction term;
- penalize candidate tangent-plane turn against that short-history direction;
- do not apply this as a direction/presence gate;
- do not add a cumulative normal/elevation term.
