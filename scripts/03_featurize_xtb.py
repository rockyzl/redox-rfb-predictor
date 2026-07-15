
#!/usr/bin/env python
"""GFN2-xTB quantum featurization of a diverse molecule subset (ALPB water).

Selects a scaffold-diverse subset stratified across the redox-potential range
(if data/subset_for_xtb.parquet is absent it is built here), then runs, per molecule:
  ETKDGv3 embed -> MMFF -> xTB GFN2 --opt loose (HOMO/LUMO/gap/dipole/energy)
  + single points on the cation/anion for vertical IP/EA.
Each molecule has a hard 180s wall-clock deadline; results checkpoint every 25.
Output: data/features_xtb.parquet
"""
import os, subprocess, tempfile, shutil, re, time, signal
import pandas as pd, numpy as np
from rdkit.Chem import MolFromSmiles, AddHs, MolToXYZBlock, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger


def build_subset(n_target=350, heavy_max=30, seed=42):
    """Diverse, potential-stratified subset from the curated table."""
    here = os.path.dirname(__file__)
    df = pd.read_parquet(os.path.join(here, "..", "data", "reddb_curated.parquet")).copy()
    df["scaffold"] = df["canonical_smiles"].apply(
        lambda s: MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s) or s)
    df["n_heavy"] = df["canonical_smiles"].apply(lambda s: MolFromSmiles(s).GetNumHeavyAtoms())
    df = df[df["n_heavy"] <= heavy_max]
    df["pb"] = pd.qcut(df["redox_potential_V"], q=7, labels=False, duplicates="drop")
    picks = [g.sample(min(n_target // df["pb"].nunique() + 1, len(g)), random_state=int(b))
             for b, g in df.groupby("pb")]
    sub = pd.concat(picks).drop_duplicates("canonical_smiles").reset_index(drop=True)
    out = os.path.join(here, "..", "data", "subset_for_xtb.parquet")
    sub.to_parquet(out, index=False)
    print(f"built subset: {len(sub)} molecules -> {out}")
    return sub
from concurrent.futures import ProcessPoolExecutor, as_completed
RDLogger.DisableLog('rdApp.*')
HARTREE_EV = 27.211386

class TO(Exception): pass
def _alarm(sig,frm): raise TO()

def sp_energy(xyz_path, wd, charge, uhf, timeout=60, threads=2):
    env=dict(os.environ,OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads))
    r=subprocess.run(["xtb",xyz_path,"--gfn","2","--alpb","water","--chrg",str(charge),"--uhf",str(uhf)],
                     cwd=wd,capture_output=True,text=True,timeout=timeout,env=env)
    m=re.search(r"TOTAL ENERGY\s+([-\d.]+)\s*Eh",r.stdout); return float(m.group(1)) if m else None

def _core(smiles, threads=2):
    m=MolFromSmiles(smiles)
    if m is None: return None
    m=AddHs(m)
    p=AllChem.ETKDGv3(); p.maxIterations=200; p.randomSeed=42
    if AllChem.EmbedMolecule(m,p)!=0:
        p2=AllChem.ETKDGv3(); p2.useRandomCoords=True; p2.maxIterations=200
        if AllChem.EmbedMolecule(m,p2)!=0: return None
    try: AllChem.MMFFOptimizeMolecule(m,maxIters=300)
    except Exception: pass
    d=tempfile.mkdtemp()
    try:
        open(os.path.join(d,"mol.xyz"),"w").write(MolToXYZBlock(m))
        env=dict(os.environ,OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads))
        r=subprocess.run(["xtb","mol.xyz","--gfn","2","--opt","loose","--alpb","water"],
                         cwd=d,capture_output=True,text=True,timeout=90,env=env)
        out=r.stdout; f={}
        g=re.search(r"HOMO-LUMO GAP\s+([-\d.]+)\s*eV",out); f["xtb_gap_eV"]=float(g.group(1)) if g else None
        homo=lumo=None
        for ln in out.splitlines():
            pt=ln.split()
            if "(HOMO)" in ln:
                for i,pp in enumerate(pt):
                    if pp=="(HOMO)": homo=float(pt[i-1])
            if "(LUMO)" in ln:
                for i,pp in enumerate(pt):
                    if pp=="(LUMO)": lumo=float(pt[i-1])
        f["xtb_homo_eV"]=homo; f["xtb_lumo_eV"]=lumo
        e=re.search(r"TOTAL ENERGY\s+([-\d.]+)\s*Eh",out); E0=float(e.group(1)) if e else None
        f["xtb_energy_neutral_Eh"]=E0
        dp=re.search(r"molecular dipole:[\s\S]*?full:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)",out)
        f["xtb_dipole_D"]=float(dp.group(1)) if dp else None
        optxyz=os.path.join(d,"xtbopt.xyz"); geo=optxyz if os.path.exists(optxyz) else os.path.join(d,"mol.xyz")
        try: Ec=sp_energy(geo,d,1,1,threads=threads)
        except Exception: Ec=None
        try: Ea=sp_energy(geo,d,-1,1,threads=threads)
        except Exception: Ea=None
        if E0 and Ec: f["xtb_vIP_eV"]=(Ec-E0)*HARTREE_EV
        if E0 and Ea: f["xtb_vEA_eV"]=(E0-Ea)*HARTREE_EV
        return f if homo is not None else None
    finally: shutil.rmtree(d,ignore_errors=True)

def worker(args):
    idx,smi=args
    signal.signal(signal.SIGALRM,_alarm); signal.alarm(180)  # hard 180s deadline
    try: r=_core(smi)
    except Exception: r=None
    finally: signal.alarm(0)
    return idx,r

if __name__=="__main__":
    _subp=os.path.join(os.path.dirname(__file__),"..","data","subset_for_xtb.parquet")
    sub=pd.read_parquet(_subp) if os.path.exists(_subp) else build_subset()
    tasks=list(zip(sub.index.tolist(), sub["canonical_smiles"].tolist()))
    results={}; done=0; t0=time.time()
    def flush():
        rows=[]
        for idx,r in results.items():
            if r: rows.append({"canonical_smiles":sub.loc[idx,"canonical_smiles"],
                               "redox_potential_V":sub.loc[idx,"redox_potential_V"],**r})
        if rows: pd.DataFrame(rows).to_parquet(os.path.join(os.path.dirname(__file__),"..","data","features_xtb.parquet"),index=False)
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(worker,t):t[0] for t in tasks}
        for fu in as_completed(futs):
            try: idx,r=fu.result(timeout=300)
            except Exception: idx,r=futs[fu],None
            results[idx]=r; done+=1
            if done%25==0:
                ok=sum(1 for v in results.values() if v)
                print(f"{done}/{len(tasks)} done, {ok} ok, {time.time()-t0:.0f}s",flush=True)
                flush()
    rows=[]
    for idx,r in results.items():
        if r: rows.append({"canonical_smiles":sub.loc[idx,"canonical_smiles"],
                           "redox_potential_V":sub.loc[idx,"redox_potential_V"],**r})
    fx=pd.DataFrame(rows); fx.to_parquet(os.path.join(os.path.dirname(__file__),"..","data","features_xtb.parquet"),index=False)
    print(f"FINAL: {len(fx)} molecules ok / {len(tasks)}, {time.time()-t0:.0f}s",flush=True)
