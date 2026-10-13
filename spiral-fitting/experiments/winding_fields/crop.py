import zarr, numpy as np, sys
R='/home/sean/Desktop/spiral_dataset/fiber_zarrs/PHercParis4-20260411134726-las-sd1-d74cca79_'
lvl=sys.argv[1]; z=int(sys.argv[2])
nx=zarr.open(R+'nx.ome.zarr/'+lvl,mode='r'); ny=zarr.open(R+'ny.ome.zarr/'+lvl,mode='r'); pr=zarr.open(R+'presence.ome.zarr/'+lvl,mode='r')
p=pr[z]; a=nx[z].astype(np.float32); b=ny[z].astype(np.float32)
print('shape',p.shape,'presence>0 frac',(p>0).mean(), 'pctl',np.percentile(p[p>0],[10,50,90,99]))
m=p>0
ax=(a-128)/127; ay=(b-128)/127; nz=np.sqrt(np.clip(1-ax*ax-ay*ay,0,1))
print('nz pctl in presence', np.percentile(nz[m],[5,25,50,75,95]))
print('raw nx at presence==0 unique-ish', np.unique(a[~m])[:5])
np.savez_compressed(sys.argv[3], p=p, nz=nz, ax=ax, ay=ay)
