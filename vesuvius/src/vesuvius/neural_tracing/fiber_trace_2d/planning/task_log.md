# Finetuning log

- Verified actualtrainer startup: optimizer_lr=[0.001], checkpoint_step21100,
  next_step21101; manager phase runninggeneration1 (exclusive startupcalibration).
  Newrun finetune_5scroll_s1a_128_sd1_16x8_ema_lr0001_20260912_223743.
  Longrunningtask leftalive; notclaiming calibrationfinished/adaptiveyet.
- Original stopped cleanly atcompletecurrent21100/2700800samples. Newlaunch
  finetune_5scroll_s1a_128_sd1_16x8_ema_lr0001-20260912_223739 root2296371.
  Configdiff independentlyverifiedonly LR/runname/resume. Existingrestoreapplies
  configLRafteroptimizerstate; no repeatwarmup. Runner tests22passed1.62s.
  Newconfig triggers normal exclusive real-update startupcalibration; afterthat
  same run becomesadaptive. No bypass or unmanagedlaunch.
- User requestedstop/resumecurrentraw at0.1xLR. OldLR0.01;new0.001.
- Approved watcher bound to originallaunch164113/root1975660, targetlastcompleted
  test2700800samples; waitsforraw/EMAserialization beforeboundedshutdown.
