"""Collect one real attention launch per configuration; preserve tensor strides."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

root=Path('/content/gliner-ab')
select('base')
captured=[]
original_attention=gliner_t4.attention
def cpu_copy(value):
    if isinstance(value,torch.Tensor):
        result=torch.empty_strided(value.size(),value.stride(),dtype=value.dtype,device='cpu')
        return result.copy_(value)
    return value
def capture(*args,**kwargs):
    if not captured:captured.append(([cpu_copy(x) for x in args],{k:cpu_copy(v) for k,v in kwargs.items()}))
    return original_attention(*args,**kwargs)
gliner_t4.attention=capture
ids=torch.tensor([tokens[i*512:(i+1)*512] for i in range(16)],device='cuda')
try:forward(fused,ids,torch.ones_like(ids))
finally:gliner_t4.attention=original_attention
assert len(captured)==1
torch.save(captured[0],root/'attention-inputs.pt')
probe='''import sys,torch
from pathlib import Path
root=Path('/content/gliner-ab')
sys.path.insert(0,str(root/'torch-ext'))
import gliner_t4
label=sys.argv[1]
if label!='base':
 name='gliner_t4_'+label
 torch.ops.load_library(str(root/label/'torch-ext'/name/(name+'.so')))
 gliner_t4.ops=getattr(torch.ops,name)
def gpu_copy(v):
 if isinstance(v,torch.Tensor):return torch.empty_strided(v.size(),v.stride(),dtype=v.dtype,device='cuda').copy_(v)
 return v
a,k=torch.load(root/'attention-inputs.pt',weights_only=False)
a=[gpu_copy(v) for v in a];k={key:gpu_copy(v) for key,v in k.items()}
with torch.inference_mode():
 result=gliner_t4.attention(*a,**k)
 torch.cuda.synchronize()
assert torch.isfinite(result).all()
'''
(root/'ncu_probe.py').write_text(probe)
ncu_results={}
for label in ('base','pos_sync','all_sync'):
    cmd=['/usr/local/cuda/bin/ncu','--set','basic','--section','SchedulerStats','--section','WarpStateStats',
         '--kernel-name','regex:disentangled_attention_v4_kernel','--launch-count','1','--target-processes','all',
         '--force-overwrite','-o',str(root/(label+'-ncu')),sys.executable,str(root/'ncu_probe.py'),label]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    text=result.stdout+result.stderr
    if result.returncode==0:
        raw=subprocess.run(['/usr/local/cuda/bin/ncu','--import',str(root/(label+'-ncu.ncu-rep')),'--page','raw','--csv'],capture_output=True,text=True,check=True)
        (root/(label+'-ncu.csv')).write_text(raw.stdout)
    (root/(label+'-ncu.txt')).write_text(text)
    ncu_results[label]={'returncode':result.returncode,'text':text}
    emit('NCU',{'variant':label,'returncode':result.returncode,'text':text[-24000:]})
    if 'ERR_NVGPUCTRPERM' in text:break
(root/'ncu-results.json').write_text(json.dumps(ncu_results,indent=2))
with zipfile.ZipFile(root/'ncu-export.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in root.iterdir():
        if p.name.endswith(('-ncu.txt','.ncu-rep')) or p.name in ('ncu-results.json','ncu_probe.py'):z.write(p,p.name)

print('T4_NCU_DONE',flush=True)
