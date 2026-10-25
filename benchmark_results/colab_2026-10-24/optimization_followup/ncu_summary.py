import csv
import io
import json
from pathlib import Path
import subprocess
import zipfile

root=Path('/content/gliner-ab')
summaries={}
for label in ('base','min3'):
    result=subprocess.run(['/usr/local/cuda/bin/ncu','--import',str(root/(label+'-ncu.ncu-rep')),'--page','raw','--csv'],
                          capture_output=True,text=True,check=True)
    (root/(label+'-ncu.csv')).write_text(result.stdout)
    reader=csv.DictReader(io.StringIO(result.stdout))
    rows=list(reader)
    wanted=('register','occupancy','warps_active','warp_issue_stalled','throughput','duration','blocks_per_multiprocessor')
    selected=[]
    if 'Metric Name' in reader.fieldnames:
        for row in rows:
            metric=row.get('Metric Name','')
            if any(s in metric.lower() for s in wanted):
                selected.append({k:row.get(k) for k in ('Metric Name','Metric Unit','Metric Value')})
    else:
        assert len(rows)>=2,'Expected unit row and kernel row'
        units,values=rows[0],rows[-1]
        for metric in reader.fieldnames:
            if any(s in metric.lower() for s in wanted):
                selected.append({'Metric Name':metric,'Metric Unit':units[metric],'Metric Value':values[metric]})
    assert selected,'No Nsight metrics parsed'
    summaries[label]=selected
    emit('NCU_METRICS',{'variant':label,'metrics':selected})
(root/'ncu-summary.json').write_text(json.dumps(summaries,indent=2))
with zipfile.ZipFile(root/'ncu-export.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in root.iterdir():
        if p.name.endswith(('-ncu.txt','-ncu.csv','.ncu-rep')) or p.name in ('ncu-results.json','ncu-summary.json','ncu_probe.py'):
            z.write(p,p.name)
