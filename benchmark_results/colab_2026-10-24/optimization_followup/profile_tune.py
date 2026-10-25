"""Profile extraction and compare existing 4/8-warp variants on one GPU."""
import ast
import collections
import cProfile
import io
import json
from pathlib import Path
import pstats
import shutil
import statistics
import subprocess
import time
import zipfile

root = Path('/content/gliner-ab')
source = (root/'benchmark.py').read_text()
exec(compile(source[:source.index('for length in (256, 384, 512, 1024):')], 'setup', 'exec'))
start = source.index('labels = {')
end = source.index("for path in ('triton','custom'):\n    extract(path, chunks)")
exec(compile(source[start:end], 'extraction_helpers', 'exec'))
tree = ast.parse(source)
normalize_node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalize')
exec(compile(ast.Module(body=[normalize_node],type_ignores=[]),'normalize','exec'))

base_ops = gliner_t4.ops
def select(path):
    cls.forward_prepared,cls.forward_packed = originals if path=='triton' else custom
    gliner_t4.VERSION = 5 if path=='custom8' else 4
    gliner_t4.ops = base_ops

out = {'gpu':torch.cuda.get_device_name(),'versions':report['versions'],
       'scope':'Same checkpoint/corpus as previous experiment; 4 CPU threads; no fast-count, no multi-stream serving.',
       'encoder':[], 'extraction':{}, 'profiles':{}, 'cprofile':{}}
emit('PROFILE_START',out)

# Compiler resource usage helps distinguish the register limit from shared-memory use.
nvcc = shutil.which('nvcc')
resources = subprocess.run([nvcc,'-arch=sm_80','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v','-c',
                            str(root/'csrc/disentangled_attention_v4.cu'),'-o',str(root/'resources.o')],
                           capture_output=True,text=True)
out['ptxas_returncode']=resources.returncode
out['ptxas']=resources.stdout+resources.stderr
emit('RESOURCES',out['ptxas'])
out['ncu_available']=shutil.which('ncu')

for length in (256,512,1024):
    for padded in (False,True):
        ids=torch.tensor([tokens[i*length:(i+1)*length] for i in range(16)],device='cuda')
        mask=torch.ones_like(ids)
        if padded:
            for i in range(16):mask[i,max(1,length-i*length//20):]=0
        select('custom4'); expected=forward(fused,ids,mask).clone()
        select('custom8'); actual=forward(fused,ids,mask)
        finite=bool(torch.isfinite(actual).all())
        assert finite,'Non-finite 8-warp output'
        times={path:[] for path in ('triton','custom4','custom8')}
        for path in times:
            select(path)
            for _ in range(3):forward(fused,ids,mask)
        for trial in range(7):
            paths=list(times)
            paths=paths[trial%3:]+paths[:trial%3]
            for path in paths:times[path].append(timed_encoder(ids,mask,path))
        row={'length':length,'padded':padded,'ms':{k:statistics.median(v) for k,v in times.items()},
             'samples':times,'eight_vs_four_equal':bool(torch.equal(expected,actual)),
             'eight_vs_four_max_abs':float((expected.float()-actual.float()).abs().max())}
        out['encoder'].append(row)
        emit('TUNE_ENCODER',row)
del plain,expected,actual
torch.cuda.empty_cache()
outputs_by_path={}
for path in ('triton','custom4','custom8'):
    extract(path,chunks)
samples={path:[] for path in ('triton','custom4','custom8')}
for trial in range(5):
    paths=list(samples)
    paths=paths[trial%3:]+paths[:trial%3]
    for path in paths:
        elapsed,outputs=extract(path,chunks)
        samples[path].append(elapsed)
        outputs_by_path[path]=outputs
        emit('TUNE_EXTRACTION',{'trial':trial,'path':path,'seconds':elapsed,'chunks_per_s':len(chunks)/elapsed})
for path,values in samples.items():
    out['extraction'][path]={'seconds':values,'chunks_per_s':len(chunks)/statistics.median(values)}
out['parity']={}
baseline=normalize(outputs_by_path['custom4'])
for path in ('triton','custom8'):
    actual=normalize(outputs_by_path[path])
    out['parity'][path+'_vs_custom4']={'identical_chunks':sum(a==b for a,b in zip(actual,baseline)),'total':len(chunks)}

def kernel_family(name):
    n=name.lower()
    if 'attention' in n or 'deberta' in n:return 'attention'
    if any(s in n for s in ('gemm','gemv','cublas','cutlass')):return 'matrix_multiply'
    if 'layer_norm' in n:return 'layer_norm'
    if 'gelu' in n:return 'gelu'
    if 'memcpy' in n:return 'memory_copy'
    return 'other'

subset=chunks[::3]
for path in ('triton','custom4','custom8'):
    extract(path,subset)
    torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        inner_elapsed,_=extract(path,subset)
    elapsed=time.perf_counter()-start
    families=collections.defaultdict(float)
    kernels=collections.defaultdict(lambda:[0,0.])
    for event in prof.events():
        if event.device_type==torch.autograd.DeviceType.CUDA:
            duration=event.time_range.elapsed_us()/1000
            families[kernel_family(event.name)]+=duration
            kernels[event.name][0]+=1;kernels[event.name][1]+=duration
    out['profiles'][path]={'chunks':len(subset),'profiled_wall_seconds':elapsed,'extraction_wall_seconds':inner_elapsed,'gpu_kernel_ms':sum(families.values()),
                            'gpu_families_ms':dict(families),
                            'top_gpu_kernels':sorted(kernels.items(),key=lambda x:x[1][1],reverse=True)[:15]}
    prof.export_chrome_trace(str(root/(path+'-trace.json')))
    emit('PROFILE',{'path':path,**out['profiles'][path]})

for path in ('triton','custom4'):
    profiler=cProfile.Profile()
    profiler.enable();elapsed,_=extract(path,subset);profiler.disable()
    stream=io.StringIO()
    stats=pstats.Stats(profiler,stream=stream).sort_stats('tottime')
    stats.print_stats(35)
    out['cprofile'][path]={'seconds':elapsed,'top_self_time':stream.getvalue()}
    (root/(path+'-cpu.txt')).write_text(stream.getvalue())
    emit('CPU_PROFILE',{'path':path,**out['cprofile'][path]})

out['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw','--format=csv'],text=True)
(root/'profile-tune.json').write_text(json.dumps(out,indent=2))
for path,outputs in outputs_by_path.items():(root/(path+'-tune-outputs.json')).write_text(json.dumps(outputs))
with zipfile.ZipFile(root/'profile-export.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in root.iterdir():
        if p.name.endswith(('-trace.json','-cpu.txt','-tune-outputs.json')) or p.name=='profile-tune.json':z.write(p,p.name)
emit('TUNE_RESULT',out)
