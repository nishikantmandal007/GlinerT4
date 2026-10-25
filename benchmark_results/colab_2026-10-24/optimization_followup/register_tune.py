"""Try two launch-bounds settings; preserve the attention math and original source."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import zipfile

root=Path('/content/gliner-ab')
cuda_source=(root/'csrc/disentangled_attention_v4.cu').read_text()
binding_source=(root/'csrc/torch_binding.cpp').read_text()
cmake_source=(root/'CMakeLists.txt').read_text()
assert cuda_source.count('8 / WARPS')==1
register_report={'variants':{},'encoder':[],'extraction':{},'parity':{}}
op_sets={'base':base_ops}
for label,target in (('min3',12),('min4',16)):
    variant=root/label
    (variant/'csrc').mkdir(parents=True,exist_ok=True)
    symbol='gliner_t4_'+label
    launch='launch_disentangled_attention_v4_'+label
    cu=cuda_source.replace('8 / WARPS',str(target)+' / WARPS').replace('launch_disentangled_attention_v4(',launch+'(')
    binding=binding_source.replace('launch_disentangled_attention_v4(',launch+'(')
    binding=binding.replace('torch::Tensor disentangled_attention(','torch::Tensor disentangled_attention_'+label+'(')
    binding=binding.replace('&disentangled_attention);','&disentangled_attention_'+label+');')
    binding=binding.replace('TORCH_LIBRARY_EXPAND(gliner_t4, ops)','TORCH_LIBRARY_EXPAND('+symbol+', ops)')
    (variant/'csrc/disentangled_attention_v4.cu').write_text(cu)
    (variant/'csrc/torch_binding.cpp').write_text(binding)
    (variant/'CMakeLists.txt').write_text(cmake_source.replace('gliner_t4',symbol))
    cmds=[['cmake','-S',str(variant),'-B',str(variant/'build'),'-DCMAKE_BUILD_TYPE=Release',
           '-DCMAKE_CUDA_ARCHITECTURES=80','-DCMAKE_PREFIX_PATH='+torch.utils.cmake_prefix_path,
           '-DCMAKE_CUDA_FLAGS=--ptxas-options=-v'],
          ['cmake','--build',str(variant/'build'),'--parallel','2']]
    logs=[]
    for cmd in cmds:
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        logs.append(result.stdout)
        if result.returncode:
            print(result.stdout,flush=True)
            raise RuntimeError('Experimental variant build failed')
    log='\n'.join(logs)
    (root/(label+'-build.txt')).write_text(log)
    library=variant/'torch-ext'/symbol/(symbol+'.so')
    assert library.exists(),str(library)
    torch.ops.load_library(str(library))
    op_sets[label]=getattr(torch.ops,symbol)
    register_report['variants'][label]={'launch_bounds_min_blocks':target//4,
        'cuda_source_sha256':hashlib.sha256(cu.encode()).hexdigest(),
        'ptxas_lines':[line for line in log.splitlines() if any(s in line for s in ('registers','spill','stack frame'))]}
    emit('REGISTER_BUILD',{'variant':label,**register_report['variants'][label]})

def select(path):
    cls.forward_prepared,cls.forward_packed=custom
    gliner_t4.VERSION=4
    gliner_t4.ops=op_sets[path]

for length in (256,512,1024):
    for padded in (False,True):
        ids=torch.tensor([tokens[i*length:(i+1)*length] for i in range(16)],device='cuda')
        mask=torch.ones_like(ids)
        if padded:
            for i in range(16):mask[i,max(1,length-i*length//20):]=0
        select('base');expected=forward(fused,ids,mask).clone()
        times={path:[] for path in op_sets}
        accuracy={}
        for path in times:
            select(path)
            actual=forward(fused,ids,mask)
            assert torch.isfinite(actual).all()
            accuracy[path]={'equal':bool(torch.equal(actual,expected)),
                            'max_abs':float((actual.float()-expected.float()).abs().max())}
            for _ in range(2):forward(fused,ids,mask)
        for trial in range(7):
            paths=list(times);paths=paths[trial%3:]+paths[:trial%3]
            for path in paths:times[path].append(timed_encoder(ids,mask,path))
        row={'length':length,'padded':padded,'ms':{k:statistics.median(v) for k,v in times.items()},
             'samples':times,'accuracy_vs_base':accuracy}
        register_report['encoder'].append(row)
        emit('REGISTER_ENCODER',row)

eligible={path for path in op_sets if all(a['accuracy_vs_base'][path]['equal'] for a in register_report['encoder'])}
register_report['bit_equal_candidates']=sorted(eligible)
for path in eligible:extract(path,chunks)
samples={path:[] for path in op_sets if path in eligible}
collected={}
for trial in range(5):
    paths=list(samples);paths=paths[trial%len(paths):]+paths[:trial%len(paths)]
    for path in paths:
        elapsed,outputs=extract(path,chunks)
        samples[path].append(elapsed);collected[path]=outputs
        emit('REGISTER_EXTRACTION',{'trial':trial,'path':path,'seconds':elapsed,'chunks_per_s':len(chunks)/elapsed})
for path,values in samples.items():register_report['extraction'][path]={'seconds':values,'chunks_per_s':len(chunks)/statistics.median(values)}

def structural(value):
    if isinstance(value,dict):return {k:structural(v) for k,v in value.items() if k not in ('confidence','score')}
    if isinstance(value,list):return [structural(v) for v in value]
    return value
for path,outputs in collected.items():
    a,b=normalize(outputs),normalize(collected['base'])
    register_report['parity'][path]={'identical_values':sum(x==y for x,y in zip(a,b)),'total_chunks':len(chunks),
                                    'identical_spans_and_values':structural(outputs)==structural(collected['base'])}
    (root/(path+'-register-outputs.json')).write_text(json.dumps(outputs))
(root/'register-tune.json').write_text(json.dumps(register_report,indent=2))
with zipfile.ZipFile(root/'register-export.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in root.iterdir():
        if p.name.endswith(('-register-outputs.json','-build.txt')) or p.name=='register-tune.json':z.write(p,p.name)
    for label in ('min3','min4'):
        z.write(root/label/'csrc/disentangled_attention_v4.cu',label+'/disentangled_attention_v4.cu')
emit('REGISTER_RESULT',register_report)
