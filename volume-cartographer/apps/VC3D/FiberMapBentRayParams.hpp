#pragma once

// Tracing parameters of the Fiber Map's bent rays (see FiberMapBentRays.hpp).
// Plain values with no library dependency, so the winding solver, the layout
// and the geometry unit share one definition. Lengths in voxels of the
// fibers' frame; the intents are what they come to at 2.4 um/voxel, converted
// by the caller like the layout's other lengths. None of these decides a
// winding order: the reading is the side of a crossing, these bound the
// tracing. The umbilicus-radius cutoff the rays also obey is the solver's own
// SolverParams::minUmbilicusRadiusVx, passed alongside, not duplicated here.
namespace vc3d::fiber_map::bent
{

// What justifies a stretch's orientation for its readings to constrain
// (see FiberMapBentRays.hpp): link witnesses only, or the umbilicus vote
// too where no witness exists.
enum class AnchorPolicy {
    // A stretch with no link witness has its readings recorded but withheld.
    LinksOnly,
    // A stretch with no link witness constrains with the vote's anchor when
    // the vote was decisive (Oriented); split or voteless stretches are
    // withheld.
    UmbilicusOrLinks,
};

struct BentRayParams {
    // |axis . e_r| below which a sample's straight (radial) reading is ill
    // conditioned and the sample belongs to a bent run (dimensionless). The
    // same gate makes a sample a voter on the fiber's orientation: every
    // ill-conditioned run is voteless and lies within one vote section.
    double conditioningGate = 0.5;
    // Ray integration step. Intent: 0.002 cm.
    double stepVx = 8.0;
    // Ray length each way; a ray is not trusted further. Intent: 0.1 cm.
    double maxLengthVx = 417.0;
    // Arclength between ray starts along a run, measured from the run's
    // canonical end. Intent: 0.002 cm.
    double spacingVx = 8.0;
    // Which anchors let a stretch's readings constrain (assembly, not
    // tracing: no cached artifact reads it).
    AnchorPolicy anchorPolicy = AnchorPolicy::LinksOnly;
    // A reading taken after the ray bent through a crease of the field -
    // its direction turned by more than creaseTurnDeg degrees since its
    // seed step while the field's axis went through tangential (the
    // smallest |axis . e_r| along the ray below creaseConditioning) - is
    // recorded and withheld: the sheet turned over along the ray and the
    // side read on the far side is not the next layer's. Measured on
    // PHerc0139: two to four readings per build, the one false ring among
    // them. creaseTurnDeg <= 0 disables it.
    double creaseTurnDeg = 60.0;
    double creaseConditioning = 0.05;
};

} // namespace vc3d::fiber_map::bent
