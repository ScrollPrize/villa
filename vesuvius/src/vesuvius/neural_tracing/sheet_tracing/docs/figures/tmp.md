starting from a polyline (can be fiber, line from initial segment, anything), at scale 1 (19.2um) and a direction

models: 
"point" cnn -- takes [30,62,30] crops, tokenize
"line"  cnn -- takes [14,14,14] crops, tokenize
"tracer" -- two transformers and a final head:
- 4 transformer blocks with axial attention along each row then each column of the line cnn tokens
- 4 transformer blocks with attention over all 5 "current" points, plus cross attention over all the line tokens
- head predicts, for each point, displacement to the current sheet, tilt, and confidence it is still on the same sheet
	
tilt is a correction to the interpolated tangent between control points, so the model can bend the polyline more than a regular interp between points 

1. at 5 evenly spaced points along the polyline (currently 60vx apart along the line) , 
    - extract an oriented crop (for the seed, this is just oriented "along" the sheet normal)
    - encode with a 3d cnn , tokenize
2.  every 10vx along the line, 
    - extract a smaller, oriented crop
    - encode with a smaller 3d cnn, tokenize
3.  advance the line "forward". forward is along a direction vector, which during tracing is just the forward direction it has been traveling, and take new crops
    - tracer transformer attends to all of the current crops tokens + the previous ones 
    - at each point predict: displacement to the sheet, tilt, confidence we are still on the same sheet
4. advance the polyline forward, respace evenly along the line, and do it again

the previous 3 rows of line tokens are kept as "memory" and continue to be attended to by the model

at inference this looks like
	- start at the seed
	- move the line forward
	- stop at the first row which has no points that meet the confidence threshold
    - remove any points which are below the confidence threshold , 
    - write out the tifxyz



