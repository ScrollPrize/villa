# Evaluate stopped finetuning run

Stop active LR0.001 combined training safely. Freeze current EMA999, currentEMA99,
currentraw,best raw; evaluate in thatorder. Do notresume training. Compare all
four with previous evaluated snapshots on eachscroll, fulltables aftereach.
Use same locked50fiber/100direction workload,20%scrollmeans, highlightedbest/ties,
no pooledoverallscore. Record checkpointages/loss/hashes and artifacts.
# Additional evaluations and presentation

After the four stopped-run snapshots, evaluate finetuning bestEMA99 and bestEMA999.
Tables identify actual snapshot sample/update counts, weight kind and finetuning;
no early/previous age labels. Overall score uses Classic-normalized geometric
relative improvement, scaled to Classic's exact arithmeticmean in mm-equivalent.
