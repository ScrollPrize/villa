#pragma once
// A NeXus scan file with an NXtomo subentry, as GDA writes it at Diamond (beamline I12): one master per scan
// (<n>.nxs) that links one frame file holding the darks, the flats and the projections of a circular scan one
// after the other, told apart by image_key (2 dark, 1 flat, 0 projection).
//
//   <entry>/tomo_entry/definition                     "NXtomo"
//   <entry>/tomo_entry/instrument/detector/data       the frames (frames, rows, columns): a link into the frame file
//   <entry>/tomo_entry/instrument/detector/image_key  per frame
//   <entry>/tomo_entry/sample/rotation_angle          per frame [deg]
//   <entry>/tomo_entry/control/data                   the ring current, one value for the scan
//   <entry>/instrument/<detector>/count_time          per frame [s]                      (GDA)
//   <entry>/instrument/zebraContinuousMoveController/CaptureTime   per frame             (GDA fly scans)
//   <entry>/before_scan/<group>/<motor>               every motor's position before the scan (GDA)
//   <entry>/before_scan/flyscan_cfg/tomography_cmd    the command, with inBeamPosition and outOfBeamPosition
//   <entry>/before_scan/flyscan_cfg/tomography_translation   the stage that moves the sample out of the beam
//
// read_record and open_scan (record.hpp, scan.hpp) hand such a file to the readers here, which give the same
// Record and Scan a BLISS master gives:
//
//   stretches     the runs of equal image_key, in the file's order, each one piece of the frame file. A record
//                 shorter than the frame file gives the frames it describes.
//   angle         rotation_angle is the angle at the start of the exposure (the Zebra's capture at the trigger);
//                 the arc swept during it is the median step x exposure / frame period, the period from
//                 CaptureTime (milliseconds or seconds: whichever fits between the entry's start and end time).
//                 Without CaptureTime no arc is stated.
//   height        the scan does not move along the axis: the height stage's position before the scan
//   across beam   the command's inBeamPosition for the projections and outOfBeamPosition for the flats (the
//                 position before the scan is a snapshot that may be taken while the stage moves); the snapshot
//                 where the command states none
//   ring current  the one value of the scan stands for each of its frames
//
// What the file does not hold is read from a supplement file beside the master, helirec_acquisition.json:
//
//   {"pixel_size_um": 7.91,                  the sample's pixel (required: nothing can be placed without it)
//    "energy_kev": 53,                       optional
//    "propagation_distance_m": 0.5,          optional: sample to detector
//    "source_sample_distance_m": 51,         optional
//    "fan_sense": -1,                        optional, 1 or -1 (default 1): the sense of the beam's divergence against
//                                            the recorded angle (Acquisition::fan); -1 at Diamond I12
//    "height_stage": "ss1_y1",               optional: the motor that sets the height of the sample in the beam
//    "stages_on_rotation": ["ss1_tx", "ss1_tz"]}   optional: the two stages that move the sample on the rotation
//
// height_stage defaults to <prefix>_y1 or <prefix>_y and stages_on_rotation to <prefix>_tx, <prefix>_tz, with
// <prefix> the translation stage's name up to its last "_" ("ss1_x" -> "ss1"), where the file holds such motors.
#include <string>

#include "helirec/calibration/record.hpp"
#include "helirec/calibration/scan.hpp"

namespace helirec::calibration::nxtomo {

inline constexpr const char* kSupplement = "helirec_acquisition.json";

// whether the HDF5 file at `path` holds an entry with an NXtomo subentry (false for anything else, unreadable files included)
bool is_master(const std::string& path);

// frame_files: also open the frame file to count its frames (FrameFiles::frames_in_file)
Record read_record(const std::string& path, bool frame_files);

// master: the absolute path of the file
Scan open_scan(const std::string& master);

}  // namespace helirec::calibration::nxtomo
