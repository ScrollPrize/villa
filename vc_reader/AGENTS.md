### about this repo
this package is intended to provide useful tools for working with the vesuvius challenge data
- metadata browsing and parsing
- volume and image data fetching and sampling
- augmentations

performance is not an afterthought for anything implemented in this repository. these tools are intended for very large scale multidimensional image processing and machine learning training. it is critical that we account for performance with any changes. 

### general rules
- for augmentations and sampling, whenever we add a cpu implementation we should have a matching cuda implementation
- in many cases, these tools will be used in pytorch dataloaders or within ml training pipelines, and this should be considered when designing things

### precision and dtypes
- any dtype higher than float32 must be justified. do not use higher dtypes just because you've seen them used elsewhere in the codebase. 

### optimization and performance
when writing code for this repository especially when doing work on performance optimization, consider the type of work you are doing:
- for augmentations, bit identical outputs are not necessary. we can tolerate some differences because we are after all changing the input image in the first place. we care more about semantic equivalence
- for some things like coordinate sampling _before_ augmentations , precision or near-equivalence may be more necessary.
- negating the upload time to gpu , if an equivalent cuda implementation comes out slower than a cpu side version, this should alert you that something is wrong. very rarely should this be the case 
- when working on performance , do not just limit yourself to making tiny hillclimb changes
- do not assume the implementations of anything or the dtypes chosen or sizes of anything are static or chosen correctly. 
- there may be implementations that are doing far more work than necessary, and it would be a mistake to optimize that rather than starting out by reducing unnecessary work

### legacy and compatibilty handling
in general, do not handle old implementations or legacy implementations. do not use compatibility shims or anything like this. do not mention old code or legacy implementations anywhere in readmes or code comments. if you think compatibility like this is necessary, you must ask before writing it. 