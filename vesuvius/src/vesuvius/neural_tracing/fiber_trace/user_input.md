# various
- cpu data loader shall only calc the relevant crops and load them from zarr not process them in any way
- label derivation etc shall all happen on gpu after batch transfer to gpu
- all the sampling should be pseudo random deterministic so we alway access the same data in the same order at training start
    - to make that work with the valid mask the dataset generation should check the validity not the data loader - it only needs to sample the center point of the tgt patch for the negative samples - and basically just skip those samples that are invalid until we have the required count of negative valid samples (note this does not depend on batch size OR crop size!)
- import relevant lasagna functions for decode etc, do not reimplement
- to make stuff determinstics (like augemetations etc. crop offset) use different random number gens by "thing" seeded/idxed by the iteration number (so each iteration will get deterministic values) 

# batch data selection
- each batch is only data from one fiber
- (3/4 of batch, rounded up by using `N - floor(N / 4)`): randomly sample cps from the randomly selected fiber (dataset size == fiber count)
- (1/4 of batch, rounded down by using `floor(N / 4)`): random volume samples - using a deterministic pseudo random dist with a (configurable current) modulus of 1000 so we always use same samples
- batch offset:
    - the pos samples should not be centered just on the cps, we want to shift eh crop center as far as possbile, just retaining a 10vx margin around the cp (this offset too should be deteministic)

# label areas
- given the normal plane from the lasagna data at the cp
- positive (this is the fiber): within +-40vx along the fiber from the rounded CP, within +-40vx in the normal plane cross-fiber direction, and +-10vx perpendicular
- neg: non-cp batch samples and samples from cp batch samples that are:
    - at least 30vx from the cp normal plane
    - AND in the 90° cone along the normal (in both direction) that originates at the line

# conditioning
- the unet should output some hidden latent layer which we interpreet through some additional head which conditions on the direction vector
    - up to 30 deg from the gt cp dir we consider still to be a positive sample that should output the same data as the gt dir
    - between 30-60 its undefined
    - from 60 to 90 it becomes negative - and the direction shall for now be ambiguous with regards to flips so afterwards it inverts again (and 180 == 0 deg)
- for now we only jitter the condition by +-30 deg keeping postitive labeled samples postive
- neg samples should just be conditioned with a random direction

# cli train reporting
- table with with one row per iteration / batch
- should be sum / of batch (dep on value) + it number
- col headers repeat every 20 rows
- explaination for shortedn col header only once at the start

# prefetch mode
flag --prefetch to run with a regular training
- it should skip all the actual work and focus on generating the relevant zarr chunk indices
- generate the full list of overall chunks with the specified config
- then run parallel download with up to 16 cunks in parallel and a progress bar and MiB/s
