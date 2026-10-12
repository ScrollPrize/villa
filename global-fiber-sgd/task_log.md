# Fresh LR and width-LR trials

Plan uv_bucket_lr_plan.md. User requests another10xstateLR withsame0.5position
loss,thenwidthfactor10x/.1x onbetterstateLRbyreferenceL1. Disclosedinterpretation:
widthoptimizerLR20/.2 versus2, not changinglossweights. Freshinitalltrials,
10k sampledpasses,full1k tables,same6exclusiveGPUs/globalbatch/seed/dataset.
Prior0.5 results committedd3b562252 inuv_bucket_half_results.md.
Configreviewapproved;9testspassed2.58s. StateLR10x approvedrunner launched
session48433,manager8a78b8b9-dbd5-457e-a9bc-0283fc51d4f2;exclusivegrantacquired.
StateLR10xfinished,suiteexit0:loss7.583289,L1=5.41051,retained44.5751%,
zero bucket signs, widthfactor.983291. CurrentstateLRbaseline loss3.706361,
L1=1.967458,retained50.1774%, so selectcurrent. Fullprogressreportedthrough10k.
IndependentreviewapprovedwidthLR20/.2 fresh configs withcurrentstateLR,
nootherchanges. Launched approvedrun_uv_bucket_width_suite.json.
Widthsuite session69270,manager5d39fc6f-0124-4db5-8957-f0172effec43.
SixGPUchecks passedforwidth20; full1kcheckpointtablesreported while running.

All3newtrialscompleted,finitearrays,bothrunnersexit0,bothmanagerleasesreleased.
State10xloss7.5833/L1=5.41051;selectedcurrentbaseline3.7064/1.96746.
Width20loss3.70455/L1=1.96271; width.2loss3.70811/L1=1.97692.
Retention50.1774%and50.1649%;zero finalbucketviolations. WidthLReffectminimal;
nochange todefaults. Times165.38s/166.18s/169.09s. Fulltablesandcommands
inuv_bucket_lr_results.md;progressprintedwithall1kcheckpointrows.
