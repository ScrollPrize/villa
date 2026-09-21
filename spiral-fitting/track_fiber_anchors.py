"""Add actual accepted fiber centerlines to track-anchor searches, never ribbons."""
import numpy as np
from pathlib import Path

from audit_track_gap_bridges import ReferenceMesh,radius_trees
from winding_arclength import source_grid


class PhysicalAnchorMesh(ReferenceMesh):
    def __init__(self,workspace):
        super().__init__(workspace)
        self.patch_faces=len(self.xyz)
        payload=np.memmap(Path(workspace).parent/self.document['geometry_file'],dtype='<f8',mode='r').reshape(-1,5)
        profile=self.document['layout'].get('arclength');groups={};xyz=[];theta=[];objects=[]
        self.excluded_fiber_segments=0
        for source in self.document['sources']:
            if not source['fiber'] or source.get('quarantined',False):continue
            if source['rows']!=3:raise ValueError('Unexpected fiber workspace representation')
            # Exporter preserves the annotated centerline in row 1. Rows 0/2
            # are synthetic display geometry and never enter this index.
            center=source_grid(payload,source)[1]
            lengths=np.linalg.norm(np.diff(center[:,:3],axis=0),axis=1)
            keep=np.flatnonzero((lengths>1e-8)&(lengths<=4.))
            self.excluded_fiber_segments+=int(len(lengths)-len(keep))
            if not len(keep):continue
            component=source['component']
            if component not in groups:
                groups[component]=len(self.sources)
                self.sources.append(dict(id=component,fiber=True))
            indices=np.column_stack((keep,keep+1,keep))
            xyz.append(center[indices,:3])
            turns=np.interp(center[:,3],profile['arclength'],profile['turns']) if profile else center[:,3]/self.document['layout']['turn_width']
            theta.append(turns[indices]);objects.append(np.full(len(keep),groups[component],int))
        if xyz:
            extra=np.concatenate(xyz);centers=extra.mean(1);radii=np.linalg.norm(extra-centers[:,None],axis=2).max(1)
            self.trees.extend(radius_trees(centers,radii,offset=len(self.xyz)))
            self.xyz=np.concatenate((self.xyz,extra));self.theta=np.concatenate((self.theta,np.concatenate(theta)))
            self.objects=np.r_[self.objects,np.concatenate(objects)]
            self.centers=np.concatenate((self.centers,centers));self.radii=np.r_[self.radii,radii]
            self.normals=np.concatenate((self.normals,np.zeros((len(extra),3))))
        # A centerline has no sheet normal. Zero normals deliberately impose
        # no tangent-plane condition on fiber contacts; patch face normals
        # still come directly from the accepted geometry.
