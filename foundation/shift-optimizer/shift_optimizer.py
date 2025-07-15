# ================================================================
#  SHIFT OPTIMIZER  ·  ESRF 17 Jul 08 → 22 Jul 08  (Europe/Rome)
# ================================================================
#  – 8‑hour shifts
#  - rest ≥16 h between consecutive shifts
#  – Non‑experts: ≥1 coached shift before any solo shift
#  – Giorgio must be on BM18 blocks starting 08:00 on 17 & 18 Jul (experiments)
#  – ID11 shift 21 Jul 08‑16 must include Giorgio AND Valentina (experiments)
#  - Papyrologists and Valentina never on night shift
#  – Valentina & Paul never alone
#  – Paul shares ≥1 shift with each of Giorgio, Stephen, Sean, Elian
#  – Personal availability, nights only for tech team, Paul ≤ 02, ≥ 16 h rest
#  – Objective: minimize:  1) max_shifts_per_person
#                          2) max_nights_per_person
#                          3) cumulative_total_shifts
#               maximize:  4) attendance to dinner with JP
# ================================================================

import datetime as dt
from itertools import combinations
from ortools.sat.python import cp_model

# ----------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------
TZ = dt.timezone(dt.timedelta(hours=2))                     # CEST

START = dt.datetime(2025, 7, 17, 8, tzinfo=TZ)
END   = dt.datetime(2025, 7, 22, 8, tzinfo=TZ)              # exclusive
ID11_START = dt.datetime(2025, 7, 21, 8, tzinfo=TZ)

DINNER_HRS = [dt.datetime(2025, 7, 17, 20, tzinfo=TZ),
              dt.datetime(2025, 7, 17, 21, tzinfo=TZ)]

people = {
    # name         expert night_ok  avail_from                        avail_to
    'Giorgio'  : (True , True ,  START,                            END),
    'Sean'     : (True , True ,  START,                            END),
    'Stephen'  : (True , True ,  START,                            END),
    'Elian'    : (False, True ,  START,                            END),
    'Valentina': (False, False, START,                            END),
    'Marzia'   : (False, False, START,  dt.datetime(2025,7,20,0,tzinfo=TZ)),
    'Gianluca' : (False, False, dt.datetime(2025,7,20,8,tzinfo=TZ), END),
    'Paul'     : (False, True ,  dt.datetime(2025,7,18,8,tzinfo=TZ),
                                                  dt.datetime(2025,7,21,0,tzinfo=TZ)),
    'JP'       : (False, False, START,    dt.datetime(2025,7,17,22,tzinfo=TZ)),
}
experts      = [p for p,(e,*_) in people.items() if e]
non_experts  = [p for p in people if p not in experts]

# Build every 8‑hour BM18 block
shifts = []                               # index → (start_dt, tag)
t = START
while t + dt.timedelta(hours=8) <= END:
    shifts.append((t, 'BM18'))
    t += dt.timedelta(hours=1)

# Append ID11 block
ID11_IDX = len(shifts)
shifts.append((ID11_START, 'ID11'))
BM18_IDX = [i for i,(_,tag) in enumerate(shifts) if tag=='BM18']

def covers(start: dt.datetime, hour: dt.datetime) -> bool:
    return start <= hour < start + dt.timedelta(hours=8)

# ----------------------------------------------------------------
# 2. Model
# ----------------------------------------------------------------
m  = cp_model.CpModel()
x  = {}                                 # (person,shift) -> Bool
exp_on = {}                             # expert present on shift

# 2·1  variables + availability
for p,(isE, night_ok, a_f, a_t) in people.items():
    for s,(st,tag) in enumerate(shifts):
        var = m.NewBoolVar(f"x_{p}_{s}")
        x[p,s] = var

        # window
        if not (a_f <= st and st+dt.timedelta(hours=8) <= a_t):
            m.Add(var == 0)

        # ID11: Giorgio & Valentina required
        if s == ID11_IDX:
            if p in ('Giorgio','Valentina'):
                m.Add(var == 1)
            continue

        # Giorgio not allowed on overlapping BM18 block
        if p=='Giorgio' and tag=='BM18':
            if st < ID11_START+dt.timedelta(hours=8) and st+dt.timedelta(hours=8) > ID11_START:
                m.Add(var == 0)

        # night bans
        night_block = any((st+dt.timedelta(hours=k)).hour < 8 for k in range(8))
        if night_block and not night_ok:
            m.Add(var == 0)

        # Paul's ≤ 02
        if p=='Paul' and (st+dt.timedelta(hours=8)).hour > 2:
            m.Add(var == 0)

# Giorgio fixed on BM18 17 & 18 Jul 08‑16
for must in (dt.datetime(2025,7,17,8,tzinfo=TZ),
             dt.datetime(2025,7,18,8,tzinfo=TZ)):
    idx = next(i for i,(st,tag) in enumerate(shifts) if tag=='BM18' and st==must)
    m.Add(x['Giorgio', idx] == 1)

# 2·2  expert flags
for s in BM18_IDX:
    f = m.NewBoolVar(f"expert_{s}")
    exp_on[s] = f
    m.AddBoolOr([x[e,s] for e in experts]).OnlyEnforceIf(f)
    m.Add(sum(x[e,s] for e in experts) == 0).OnlyEnforceIf(f.Not())

# 2·3  coached‑before‑solo for non‑experts
chron = sorted(BM18_IDX, key=lambda i: shifts[i][0])
for p in non_experts:
    trained = m.NewBoolVar(f"trained_prev_{p}_init")
    m.Add(trained == 0)
    coached_any = []
    for s in chron:
        coached = m.NewBoolVar(f"coached_{p}_{s}")
        m.Add(coached <= x[p,s])
        m.Add(coached <= exp_on[s])
        m.Add(coached >= x[p,s] + exp_on[s] - 1)
        coached_any.append(coached)

        solo = m.NewBoolVar(f"solo_{p}_{s}")
        m.Add(solo <= x[p,s])
        m.Add(solo <= 1 - exp_on[s])
        m.Add(solo >= x[p,s] + (1 - exp_on[s]) - 1)
        m.Add(trained == 1).OnlyEnforceIf(solo)

        next_tr = m.NewBoolVar(f"trained_prev_{p}_{s}")
        m.AddBoolOr([trained, coached]).OnlyEnforceIf(next_tr)
        m.AddBoolAnd([trained.Not(), coached.Not()]).OnlyEnforceIf(next_tr.Not())
        trained = next_tr
    m.Add(sum(coached_any) >= 1)

# 2·4 coverage ≥1 per hour
hours = [START + dt.timedelta(hours=h)
         for h in range(int((END-START).total_seconds()//3600))]
for h in hours:
    crew = [x[p,s] for p in people
            for s,(st,tag) in enumerate(shifts) if tag=='BM18' and covers(st,h)]
    m.Add(sum(crew) >= 1)

# 2·4‑b never‑alone rule for Valentina & Paul
for s in BM18_IDX + [ID11_IDX]:
    size = m.NewIntVar(0,len(people),f"size_{s}")
    m.Add(size == sum(x[p,s] for p in people))
    for p_alone in ('Valentina','Paul'):
        m.Add(size >= 2).OnlyEnforceIf(x[p_alone,s])

# 2·4‑c Paul pairs with each specified buddy ≥1 time
for buddy in ('Giorgio','Stephen','Sean','Elian'):
    meet = []
    for s in BM18_IDX + [ID11_IDX]:
        both = m.NewBoolVar(f"Paul_with_{buddy}_{s}")
        m.Add(both <= x['Paul',s])
        m.Add(both <= x[buddy,s])
        m.Add(both >= x['Paul',s] + x[buddy,s] - 1)
        meet.append(both)
    m.Add(sum(meet) >= 1)

# 2·5 rest ≥16 h
for p in people:
    idx = BM18_IDX + ([ID11_IDX] if p in ('Giorgio','Valentina') else [])
    for a,b in combinations(idx,2):
        if abs((shifts[a][0]-shifts[b][0]).total_seconds()) < 24*3600:
            m.Add(x[p,a] + x[p,b] <= 1)

# 2·6 dinner attendance
dinner_free = {}
for p,(_,_,a_f,a_t) in people.items():
    if not all(a_f <= h < a_t for h in DINNER_HRS):
        continue
    busy = [x[p,s] for s in BM18_IDX
            if any(covers(shifts[s][0],h) for h in DINNER_HRS)]
    free = m.NewBoolVar(f"dinner_{p}")
    dinner_free[p] = free
    m.Add(sum(busy)==0).OnlyEnforceIf(free)
    for b in busy:
        m.AddImplication(b, free.Not())

# ----------------------------------------------------------------
# 3. Objective – minimise max_shifts then max_nights …
# ----------------------------------------------------------------
shift_cnt, night_cnt = {}, {}
max_shifts = m.NewIntVar(0,len(BM18_IDX),"max_shifts")
max_nights = m.NewIntVar(0,len(BM18_IDX),"max_nights")

for p in people:
    idx = BM18_IDX + ([ID11_IDX] if p in ('Giorgio','Valentina') else [])
    # total shifts
    tot = m.NewIntVar(0,len(idx),f"cnt_{p}")
    m.Add(tot == sum(x[p,s] for s in idx))
    shift_cnt[p] = tot
    m.Add(tot <= max_shifts)
    # night shifts = blocks with start.hour < 8
    n_idx = [s for s in idx if shifts[s][0].hour < 8]
    ncnt = m.NewIntVar(0,len(n_idx),f"night_{p}")
    m.Add(ncnt == sum(x[p,s] for s in n_idx))
    night_cnt[p] = ncnt
    m.Add(ncnt <= max_nights)

total_shifts = sum(shift_cnt.values())
m.Minimize(max_shifts * 10000              # 1) fairness overall
           + max_nights * 100              # 2) fairness on nights
           + total_shifts * 10             # 3) small
           - sum(dinner_free.values()))    # 4) maximise dinner

# ----------------------------------------------------------------
# 4. Solve
# ----------------------------------------------------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60
solver.parameters.num_search_workers = 8
status = solver.Solve(m)
assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), "No feasible rota"

# ----------------------------------------------------------------
# 5. Output
# ----------------------------------------------------------------
print("\n***  FINAL ROTA  ***\n")
rota = []
for s,(st,tag) in enumerate(shifts):
    crew = [p for p in people if solver.Value(x[p,s])]
    if crew:
        rota.append((st,tag,crew))
rota.sort()
for st,tag,crew in rota:
    end = st + dt.timedelta(hours=8)
    txt = ", ".join(f"{p}{' (E)' if p in experts else ''}" for p in crew)
    print(f"{tag:4}  {st:%d %b %H:%M} → {end:%d %b %H:%M}   {txt}")

att = [p for p in dinner_free if solver.Value(dinner_free[p])]
print("\nDinner attendance 17 Jul 20‑22:", att)
print("Maximum total shifts:", solver.Value(max_shifts))
print("Maximum night shifts:", solver.Value(max_nights))
for p in people:
    print(f"  {p:10}: {solver.Value(shift_cnt[p])} shifts"
          f"  |  {solver.Value(night_cnt[p])} night(s)")
