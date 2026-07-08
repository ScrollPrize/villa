now lets implement all the augmentations.
For testing lets do an augment mode that will load just the center strip-patch and apply all possible augmentations to it (only once per sample) and save that as contact sheet for vis.
one sample in the sheet should be not-augmetnatioed
show random combined augmentations as their own all-random contact-sheet row.
all geometric augmentations should happen on the coords (so how coords are selected from the strip) so the sampling is still only once from the zarr
geometric augmentations should use an oversized strip-coordinate area so the final augmented patch is derived from coords without edge artifacts or image reinterpolation artifacts
all augmentations that happen after loading from zarr should be gpu based for now this means the image based augmentations
the contact sheets should visualize the fiber line as a drawn line at 50 percent opacity for example via cv2; normally this is straight but geometric distortion augmentations should distort it consistently
augmentation visualization should show extreme min and max examples for each augmentation plus random combined examples, giving three contact-sheet rows: min, max, all-random
