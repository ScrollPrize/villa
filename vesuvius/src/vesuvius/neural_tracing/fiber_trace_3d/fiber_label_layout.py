"""Deterministic screen-space diagnostic label placement."""
import math


def place_labels(items, project, bounds, measure):
    grid={};seen=set();result=[];cell=32;directions={}
    def cells(box):
        x,y,r,b=box
        return [(i,j) for i in range(math.floor(x/cell),math.floor(r/cell)+1)
                for j in range(math.floor(y/cell),math.floor(b/cell)+1)]
    for item in items:
        anchor=project(item['uv']);key=(*anchor,item['text'])
        if key in seen:continue
        seen.add(key)
        w,h=measure(item['text']);w+=2;h+=2
        peer=project(item['peer']) if 'peer' in item else anchor
        preferred=(1 if anchor[0]>peer[0] else -1,1 if anchor[1]>peer[1] else -1)
        if anchor==peer:preferred=(-1,-1) if item.get('family',0)==0 else (1,1)
        quadrants=[preferred,(-preferred[0],preferred[1]),(preferred[0],-preferred[1]),(-preferred[0],-preferred[1])]
        quadrants=[(dx,dy) for dx,dy in quadrants
                   if (anchor[0]+3+w<=bounds[2] if dx>0 else anchor[0]-3-w>=bounds[0])
                   and (anchor[1]+3+h<=bounds[3] if dy>0 else anchor[1]-3-h>=bounds[1])]
        if not quadrants:raise ValueError('No space for diagnostic label at image boundary')
        used=directions.setdefault((round(anchor[0]),round(anchor[1])),{})
        minimum=min(used.get(q,0) for q in quadrants)
        quadrants=[q for q in quadrants if used.get(q,0)==minimum]
        chosen=None
        for radius in range(3,int(max(bounds[2]-bounds[0],bounds[3]-bounds[1]))+32,8):
            for dx,dy in quadrants:
                x=anchor[0]+radius if dx>0 else anchor[0]-radius-w
                y=anchor[1]+radius if dy>0 else anchor[1]-radius-h
                box=(x,y,x+w,y+h)
                if x<bounds[0] or y<bounds[1] or box[2]>bounds[2] or box[3]>bounds[3]:continue
                neighbors={n for c in cells(box) for n in grid.get(c,[])}
                if any(x<result[n]['box'][2] and box[2]>result[n]['box'][0] and
                       y<result[n]['box'][3] and box[3]>result[n]['box'][1] for n in neighbors):continue
                chosen=box;used[(dx,dy)]=used.get((dx,dy),0)+1;break
            if chosen is not None:break
        if chosen is None:raise ValueError('Cannot place diagnostic labels without overlap; increase image size')
        n=len(result)
        result.append(dict(item=item,anchor=anchor,box=chosen))
        for c in cells(chosen):grid.setdefault(c,[]).append(n)
    return result
