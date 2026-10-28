"""Freeze bounded, feature-grouped numerical panels from exact OpenML sources."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {44140:'development',44144:'development',44148:'held_dataset',45034:'held_dataset'}

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def signature(values):
    row=np.asarray(values,dtype='<f8').copy()
    row[np.isnan(row)]=np.nan
    # Signed zero has no predictive distinction and must not split a duplicate group.
    row[row==0]=0.0
    return digest(row.tobytes())

def make_panel(raw, description):
    array,_=arff.loadarff(io.StringIO(raw.decode('utf-8')))
    source=pd.DataFrame(array)
    target=description['default_target_attribute']
    ignored=description.get('ignore_attribute',[])
    if isinstance(ignored,str):ignored=[ignored]
    ignored=list(ignored or [])
    if description.get('row_id_attribute'):ignored.append(description['row_id_attribute'])
    ignored += [c for c in source if c.casefold() in ['id','date']]
    features=[c for c in source if c!=target and c not in ignored]
    assert len(features)>0 and target not in features
    numeric=source[features+[target]].apply(pd.to_numeric,errors='raise')
    assert not np.isinf(numeric[features].to_numpy(dtype=float)).any()
    valid=numeric.loc[np.isfinite(numeric[target])].copy()
    valid.insert(0,'source_row',valid.index.astype(int))
    before=len(valid)
    valid=valid.drop_duplicates(subset=features+[target],keep='first')
    valid.insert(1,'group_id',[signature(row) for row in valid[features].to_numpy(dtype=float)])
    sizes=valid.groupby('group_id',sort=True).size()
    eligible=sizes.loc[sizes.cumsum()<=6000].index
    panel=valid.loc[valid['group_id'].isin(eligible)].sort_values(['group_id','source_row']).reset_index(drop=True)
    assert 0<len(panel)<=6000
    return panel,features,{'source_rows':len(source),'finite_target_rows':before,'exact_full_row_duplicates_removed':before-len(valid),'eligible_distinct_rows':len(valid),'selected_rows':len(panel),'selected_feature_groups':panel['group_id'].nunique(),'selected_feature_missing_values':int(panel[features].isna().sum().sum()),'feature_count':len(features),'excluded_attributes':sorted(set(ignored))}

def main():
    records=[]
    for dataset,role in DATASETS.items():
        meta_path=ROOT/f'data/metadata/{dataset}.json'
        description=json.loads(meta_path.read_bytes())['data_set_description']
        assert description['status']=='active'
        raw_path=ROOT/f'data/raw/{dataset}.arff'
        if raw_path.exists():
            raw=raw_path.read_bytes();retrieval='reused exact local download'
        else:
            raw=urllib.request.urlopen(description['url'],timeout=120).read()
            raw_path.parent.mkdir(parents=True,exist_ok=True);raw_path.write_bytes(raw)
            retrieval=datetime.now(timezone.utc).isoformat()
        panel,features,quality=make_panel(raw,description)
        text=panel.to_csv(index=False,float_format='%.17g',lineterminator='\n')
        path=ROOT/f'data/panels/{dataset}.csv.gz';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(gzip.compress(text.encode('utf-8'),mtime=0))
        records.append({'id':dataset,'name':description['name'],'version':description['version'],'role':role,'declared_license':description['licence'],'target':description['default_target_attribute'],'features':features,'source_url':description['url'],'retrieved_utc':retrieval,'raw_sha256':digest(raw),'raw_bytes':len(raw),'metadata_sha256':digest(meta_path.read_bytes()),'panel_file':path.relative_to(ROOT).as_posix(),'panel_sha256':digest(path.read_bytes()),'quality':quality})
        print(json.dumps({'dataset':dataset,'name':description['name'],**quality}),flush=True)
    (ROOT/'data/manifest.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    (ROOT/'results/source_quality.json').write_text(json.dumps({'status':'PASS_BOUNDED_SOURCE_PANEL','datasets':{str(r['id']):r['quality'] for r in records},'scope':'Benchmark-transformed numerical targets; feature-grouped random partitions do not establish time/entity deployment transfer'},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
