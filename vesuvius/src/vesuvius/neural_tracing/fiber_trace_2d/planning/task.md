# Task: Fiberlet-centered replay strips and indexed overview JPEGs

Extend the full-fiber replay visualization with a second top/side CT strip pair
whose line-view geometry follows the new fiberlet trace, so its refinement can
be inspected in the fiberlet frame as well as in the existing reference frame.

Wrap long selected intervals into vertically stacked four-strip blocks in the
same image. If all complete blocks cannot fit below the JPEG limit, continue in
indexed JPEG parts so no output image dimension exceeds 65,000 pixels.
