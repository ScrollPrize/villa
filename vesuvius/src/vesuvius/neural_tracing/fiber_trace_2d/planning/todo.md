# todos

- implement/update training with the current augmentation as tested with the augment vis
    - apply the full randomized augemtations per patch
    - update config and code so we run with a batch size of 64 which should be four randomly selected cps
    - patches should be again sampled along the z axis of the strip (as already implemented I believe)
    - add tensorboard logging of batch img with an overlay of the estimated direction along the gt line
    - gt direction should be calculated from the augmented strip line
    - we only want to supervise around the control point location (rounded so 8 samples)
    - make sure dir stays lasagna ambiguous according to our representation (we cant estimate fw vs bw so like a normal the dir is ambiguous in this regard)
