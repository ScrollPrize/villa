now lets implement all the augmentations.
For testing lets do an augment mode that will load just the center strip-patch and apply all possible augmentations to it (only once per sample) and save that as contact sheet for vis.
one sample in the sheet should be not-augmetnatioed
and add a second half of the contact sheet should be all augmenations applied as they would be used together.
all geometric augmentations should happen on the coords (so how coords are selected from the strip) so the sampling is still only once from the zarr
all augmentations that happen after loading from zarr should be gpu based for now this means the image based augmentations
the contact sheets should visualize the fiber line as a drawn line at 50 percent opacity for example via cv2; normally this is straight but geometric distortion augmentations should distort it consistently
