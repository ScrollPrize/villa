import threading
import iso20_tiled as tiled

def test_core_grid_and_overlap():
    tiles=tiled.tiles_for_bounds([0,0,0],[2048,2048,2048])
    assert len(tiles)==8
    a=tiles[0];b=tiles[4]
    assert a['core_max']==[1024]*3 and a['halo_max']==[1088]*3
    assert b['halo_min'][0]==960
    assert a['halo_max'][0]-b['halo_min'][0]==128
    assert sum((t['core_max'][0]-t['core_min'][0])*(t['core_max'][1]-t['core_min'][1])*
               (t['core_max'][2]-t['core_min'][2]) for t in tiles)==2048**3
    assert tiled.tiles_for_bounds([10,20,30],[1035,25,40])[-1]['core_max']==[1035,25,40]

def test_tile_workers_are_parallel():
    gate=threading.Barrier(2);results={}
    def worker(i,t):gate.wait(timeout=5);return t
    tiled.run_tiles([1,2],worker,2,lambda i,r:results.update({i:r}))
    assert results=={0:1,1:2}

def test_only_bounds_workers_and_cost_capture_change(monkeypatch,tmp_path):
    monkeypatch.setattr(tiled.shared,'OUT',tmp_path)
    template={'commands':[{'args':['trace','-','--bbox','0','0','0','10','10','10',
                                   '--output','old','--threads','128','--beam-width','8']} ]}
    tile=tiled.tiles_for_bounds([0,0,0],[2048,256,256])[0]
    args=tiled.trace_config(template,'tile',tile,64)['commands'][0]['args']
    assert args[3:9]==list(map(str,tile['halo_min']+tile['halo_max']))
    assert '--capture-step-costs' in args and '--cached-anchors-only' in args
    assert args[args.index('--beam-width')+1]=='8'
