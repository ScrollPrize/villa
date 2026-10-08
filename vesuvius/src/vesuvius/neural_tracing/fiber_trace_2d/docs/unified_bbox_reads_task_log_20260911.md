# Archived task log: unified native bbox reads — 2026-09-10/11

- User requests removal (not compatibility alias) of sample_zyx_block and fixes
  to both implementations/callers. Exact scale must exist. Follow-up explicitly
  requires declared fill value outside volume, not zero. Mask false outside,
  true for in-volume fill/missing chunks; fetch/decode/unresolved errors throw.
- Previous profiling log archived in docs/live_inference_reader_stalls_task_log_20260910.md.
- Plan: one blocking read_zyx implementation, read_xyz coordinate adapter, optional
  mask/stats; original integer dtype; release GIL during allocation/read. Remove
  missing-level fallback options from these Python bbox APIs. C++ general Volume
  resampling policies and arbitrary coordinate sampling remain separately scoped.
- Existing source dtypes uint8/uint16 preserved. Direct inference uses full bbox
  read without mask/stats or training float adapter, including fill at boundaries.
- Planreviewpassed. Importantdiscovery: oldinference partialtiles used reflect
  padding, notzero; usernotified and declared-fill boundaryread is intentional
  behaviorchange. Separate trainingaugmentationpadding remainsunchanged.
- Removed duplicateC++ bboxhelpers and binding;131added/364removedlines atfirst
  implementation. Exactdecodedpayloadsize/errorchecks, logicaledgeclip (notjust
  chunkextent), overflowchecks and optionalauxiliarywork implemented.
- CMake existing /tmp/vc-live-native-cmake configured VC_BUILD_PYTHON=ON using
  cachednanobind_DIR=/home/hendrik/.cache/uv/archive-v0/rcrHRTTfYN8hOlRV/nanobind/cmake.
  Initial configurefailed missingnanobindpackagepath; no dependencyinstallneeded.
  cmake --build /tmp/vc-live-native-cmake --target vc_volume -j8 passed18steps.
- First15testlaunchpickedoldinstallededitableextension despitepythonpathoverride:
  12failed/3passed (oldsignature); notnewimplementationvalidation. Approvedlocal
  no-deps editablepackagereinstall launched to update actualruntimebinding.
  Newtestscoverdtype,nonzerofill,logicaledge mask,auxcombinations,missing/allfill,
  exactlevels,corruptpayload,removedAPI anddirectinferencewithoutconversions.
- Localpackage rebuild/reinstall completed43.89s; editableorigin correctedfrom
  /home/hendrik/villa/volume-cartographer to /home/hendrik/villa2/volume-cartographer.
  InstalledbindingconfirmsoldAPIabsent. 76tests passed25.83s; trainingselection
  initiallylackedvesuvius/testsimportroot, correctedcommandpassed3tests4.44s.
- Independentcodereviewpassed. Addedconcurrent64reads/sharedVolume,extremeint32
  offsets andoversizedpayloadtests: expandednative+profiling24passed1.77s.
  NullData/MissQueuedbranchescode-reviewedbutnotdirectlyinjectedthroughfixtures.
  Profilecontrolsnowmatchfullbboxforrawvswrapper; mask-enabledallocationcontrol
  retainedforcomparisonwithold256MiBsetup. Sourceconfigs/specs/docsupdated.
- GPUprofilerblockedbeforework: globalmanagerabsent. Same112-corestartrefused
  because externaljournalnowrecords224cores; psutilphysical112/logical224.
  Allsavedtasks terminal, GPUsidle. Askedapprovaltoreconcile savedCPUinventory;
  nojournalmutationorunmanagedfallback. PostchangeGPUthroughputnotyetmeasured.
- Existingnativewrapperpassed62tracer cases,1cache-root case,66Python replay/shared
  tests29.89s. Profilediagnosticflagrestoredfalse. Docs/changelogupdated; no commit
  requested. Coreimplementationcomplete; GPUcomparisonpendingmanagerapproval.
- 2026-09-11 resumedprofiling: runningmanager nowcorrectlyreports224logicalCPUs;
  /home/hendrik/villa managercodeuseslogical=True, thischeckout'sservercodeisolder.
  Do not rewriteinventory. New blocker: taskb3810842-a5e5-428e-b61f-0c0c9ae90354
  (fiber-memory-b1) retainsall8GPUgrants inrelease/orphaned state. Read-onlycheck
  ofeveryregisteredPIDfoundonlymanagerbootstrapPID3394913alive; taskowner3394900
  andotherregisteredprocessesaregone. Managerwasrecordedasclientdescendant,
  soitsownlivenesspreventsautomaticprocess_tree_deadrelease. GPUsidle.
- Installedvc.volumeextensionhasalsochangedbacktotheoldread_zyx(missing_policy)
  signature andsample_zyx_blockispresentagain; benchmarkwouldnotmeasureourfix.
  NoGPUrun,journaledit,managerterminationorreinstallperformedinthisresume.
  Needpermissionforservicerestart/recovery andrebuiltbindingselection; prefer
  isolatedbindingselectionratherthanoverwritingothershared-environmentwork.
