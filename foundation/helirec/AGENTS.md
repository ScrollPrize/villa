### about this repo
this package is intended as a c++ first app with python bindings available. behavior that is not python-specific should exist both in c++ and exposed to python via bindings.

performance must be prioritized for anything implemented in this repository. these tools are intended for very large scale multidimensional image processing. it is critical that we account for performance with any changes. 

### general rules
- for other operations that naturally work well on cuda devices, whenever we add a cpu implementation we should have a matching cuda implementation
- readme files and other repository files must not contain development notes unless they are relevant to the implementation (nonstandard implementation or gotcha that would be helpful to future developers). during development you are welcome to keep notes in a gitignored development_notes folder. 
- you can access a nvidia-gpu on most development machines by elevating the request
- before adding anything to this package, first ensure that nothing which closely matches this behavior does not already exist. prefer to extend existing implementations than to create duplicates which only differ slightly
- build times can be quite slow. you shuold use incremental builds during development (ie: `ninja -C build`), we do not need to create a new build dir for every single change 

### c++ first, python binds
- for the most part, code shuold be implemented in c++ and available to c++ callers, and python binds that code. new functionality goes into c++ first.

### cuda implementations
- we have both cpu and cuda backends. for most things within that operate on volumes or images, wherever a cpu implementation exists there should be a cuda implementation as well 

### optimization and performance
when writing code for this repository especially when doing work on performance optimization, consider the type of work you are doing:
- negating the upload time to gpu , if an equivalent cuda implementation comes out slower than a cpu side version, this should alert you that something is wrong. very rarely should this be the case 
- when working on performance , do not just limit yourself to making tiny hillclimb changes
- do not assume the implementations of anything or the dtypes chosen or sizes of anything are static or chosen correctly. 
- there may be implementations that are doing far more work than necessary, and it would be a mistake to optimize that rather than starting out by reducing unnecessary work

