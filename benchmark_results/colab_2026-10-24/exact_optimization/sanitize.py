"""Check bounds and shared-memory races on small real interface cases."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

root=Path('/content/gliner-ab')
probe=r'''import sys,torch
from pathlib import Path
root=Path('/content/gliner-ab')
sys.path.insert(0,str(root/'torch-ext'))
import gliner_t4
label=sys.argv[1]
symbol='gliner_t4_'+label
torch.ops.load_library(str(root/label/'torch-ext'/symbol/(symbol+'.so')))
candidate=getattr(torch.ops,symbol)
args=torch.load(root/'sanitizer-inputs.pt',weights_only=False)
args=[x.cuda() if isinstance(x,torch.Tensor) else x for x in args]
def run(a):
 with torch.inference_mode():
  expected=gliner_t4.ops.disentangled_attention(*a)
  actual=candidate.disentangled_attention(*a)
  torch.cuda.synchronize()
  assert torch.equal(expected.contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8))
run(args)
a=list(args);a[7]=None;a[10]=None;run(a)
a=list(args);a[7]=torch.zeros_like(args[7]);a[10]=torch.zeros_like(args[10]);run(a)
a=list(args)
a[:3]=[torch.randn((99,2,64),device='cuda',dtype=torch.float16).permute(1,0,2) for _ in range(3)]
a[7]=None;a[10]=None;a[12]=torch.tensor([0,1,34,99],device='cuda',dtype=torch.int32);a[13]=65
run(a)
print('PROBE_BYTE_EQUAL',label,flush=True)
'''
(root/'sanitizer_probe.py').write_text(probe)
tool=shutil.which('compute-sanitizer')
results={'tool':tool,'checks':{}}
if tool:
    for label in ('async_simple','async_pipeline'):
        if not (root/label).exists():continue
        for mode in ('memcheck','racecheck'):
            cmd=[tool,'--tool',mode,'--error-exitcode','99',sys.executable,str(root/'sanitizer_probe.py'),label]
            try:
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=150)
                text=result.stdout+result.stderr
                rc=result.returncode
            except subprocess.TimeoutExpired as error:
                text='Timed out after 150 seconds';rc=124
            clean=rc==0 and ('ERROR SUMMARY: 0 errors' in text or 'RACECHECK SUMMARY: 0 hazards' in text)
            (root/(label+'-'+mode+'.txt')).write_text(text)
            results['checks'][label+'/'+mode]={'returncode':rc,'clean':clean,'log':text}
            emit('SANITIZER',{'variant':label,'mode':mode,'returncode':rc,'clean':clean,'tail':text[-3500:]})
(root/'sanitizer-result.json').write_text(json.dumps(results,indent=2))
with zipfile.ZipFile(root/'sanitizer-export.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in root.iterdir():
        if p.name.endswith(('-memcheck.txt','-racecheck.txt')) or p.name in ('sanitizer-result.json','sanitizer_probe.py'):z.write(p,p.name)
