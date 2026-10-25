"""Check outputs against corpus truth, record kernel dispatch, and export evidence."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata
import zipfile
from huggingface_hub import try_to_load_from_cache

indexed = sorted([(s,i) for i,d in enumerate(docs) for s in d['chunks']], key=lambda p:len(p[0]))
assert [s for s,_ in indexed] == chunks
def norm(value, street=False):
    value = unicodedata.normalize('NFKC',value).casefold()
    if street: value = re.sub(r'\b\d+[a-z]?\b',' ',value)
    return ' '.join(re.sub(r'[^\w\s]',' ',value).split())
def matches(a,b,label):
    a,b = norm(a,label=='Street'),norm(b,label=='Street')
    return bool(a and b) and (a==b or label=='Street' and (a in b or b in a))
def score(values):
    per_doc = [collections.defaultdict(set) for _ in docs]
    for (_,doc),result in zip(indexed,values):
        for label,found in result.items():per_doc[doc][label].update(found)
    metrics = {}
    for label in labels:
        truth=hits=predicted=correct=0
        for doc,found in zip(docs,per_doc):
            expected=set(doc['truth'].get(label,[])); actual=found.get(label,set())
            truth+=len(expected); predicted+=len(actual)
            hits+=sum(any(matches(a,b,label) for b in actual) for a in expected)
            correct+=sum(any(matches(a,b,label) for b in expected) for a in actual)
        metrics[label]={'truth':truth,'recognized_truth':hits,'returned':predicted,'correct_returned':correct,
                        'recall':hits/truth if truth else None,'precision':correct/predicted if predicted else None}
    return metrics
report['ground_truth_method']='Per-document unique values; Unicode/case/space normalization; Street ignores house numbers and permits containment. Raw grouped GLiNER2 outputs, without ai-utils span resolver.'
report['ground_truth']={'triton':score(native_values),'custom':score(custom_values)}
cached=try_to_load_from_cache(name,'model.safetensors')
if isinstance(cached,str):report['model_snapshot']=Path(cached).parent.name
report['kernel_source_sha256']=hashlib.sha256((ROOT/'csrc/disentangled_attention_v4.cu').read_bytes()).hexdigest()
report['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw','--format=csv'],text=True)
dispatch={}
ids=torch.tensor([tokens[i*512:(i+1)*512] for i in range(16)],device='cuda')
mask=torch.ones_like(ids)
for path in ('triton','custom'):
    select(path)
    forward(fused,ids,mask)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        forward(fused,ids,mask)
        torch.cuda.synchronize()
    names=sorted({e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA})
    assert names,'CUDA profiling captured no kernels'
    dispatch[path]=[n for n in names if any(s in n.lower() for s in ('attention','deberta','v4'))]
    if path=='triton':assert any('deberta' in n.lower() for n in names),'Triton kernel not observed'
report['profiled_attention_kernels']=dispatch
(ROOT/'result.json').write_text(json.dumps(report,indent=2))
with zipfile.ZipFile(ROOT/'export.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for filename in ('result.json','triton_outputs.json','custom_outputs.json','benchmark.py','CMakeLists.txt'):
        archive.write(ROOT/filename,filename)
emit('VERIFIED',{'ground_truth':report['ground_truth'],'dispatch':dispatch,'model_snapshot':report.get('model_snapshot'),'export':str(ROOT/'export.zip')})
