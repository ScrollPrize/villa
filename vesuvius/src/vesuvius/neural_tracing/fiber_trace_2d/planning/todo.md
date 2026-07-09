# todos

- update prefetch for parallel and transparent chunk fetching
    - add a function to the vc3d sampling which allows us to instead of sampling (blocking and downloading) the remote volume gives us a list of chunks that are required as well as the chunks remote and cache path, so we can handle the downloading ourselves
    - the function should output all chunks we'll do the local path checking and deduplication ourselves
    - make the sample coord generation to chunk mapping run in parllalel to the fetching downloading loop - indeed we should also run several coord generators in parallel to get full cpu utilization
    - download should be done python side - and if there are connection problems/interruptions with a retry up to 10 minutes (we might loose connections some times)
    - but chunks that don't exist should not be downloaded - missing chunks are possible in zarr (Q: how does the cache indicate those?)
    - remove the dbg outputs - the progress should track:
        - how many samples we processed and got chunk information for
        - how many chunks we have queued to be downloaded (and how many we already downloaded)
        - how many chunks were hits (already in the cache)
        - an eta based on the download rate and hit ratio - assuming if the sampler loop is not done that the hit rate will stay the same as what we have seen so far
