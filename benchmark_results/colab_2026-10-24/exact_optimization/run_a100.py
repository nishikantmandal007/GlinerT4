import base64,gzip
from pathlib import Path
Path("/content/gliner-ab/async-variants.json").write_bytes(gzip.decompress(base64.b64decode('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')))
"""Accuracy-preserving experiments: strict byte and full-result equality, no tolerances."""
import ast
import functools
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import types
import zipfile

root=Path('/content/gliner-ab')
source=(root/'benchmark.py').read_text()
setup=ast.parse(source[:source.index('for length in (256, 384, 512, 1024):')])
setup.body=[n for n in setup.body if not (
    isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='plain' for t in n.targets)
    or isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
       and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='plain')]
exec(compile(setup,'setup','exec'))
start=source.index('labels = {')
end=source.index("for path in ('triton','custom'):\n    extract(path, chunks)")
exec(compile(source[start:end],'helpers','exec'))
base_ops=gliner_t4.ops
variant_sources=json.loads((root/'async-variants.json').read_text())
baseline_cuda=(root/'csrc/disentangled_attention_v4.cu').read_text()
variant_sources['min3']=baseline_cuda.replace('8 / WARPS','12 / WARPS')
op_sets={'base':base_ops}
experiment={'gate':'Strict raw byte equality for attention/encoder; complete extraction JSON including scores, spans, ordering must match its own frozen baseline.',
            'gpu':torch.cuda.get_device_name(),'versions':report['versions'],'builds':{},'raw_checks':{},
            'encoder':[],'extraction':{},'parity':{},'multilingual_parity':{}}

for label,cu in variant_sources.items():
    folder=root/label;(folder/'csrc').mkdir(parents=True,exist_ok=True)
    symbol='gliner_t4_'+label;launch='launch_disentangled_attention_v4_'+label
    cu=cu.replace('launch_disentangled_attention_v4(',launch+'(')
    binding=(root/'csrc/torch_binding.cpp').read_text().replace('launch_disentangled_attention_v4(',launch+'(')
    binding=binding.replace('torch::Tensor disentangled_attention(','torch::Tensor disentangled_attention_'+label+'(')
    binding=binding.replace('&disentangled_attention);','&disentangled_attention_'+label+');')
    binding=binding.replace('TORCH_LIBRARY_EXPAND(gliner_t4, ops)','TORCH_LIBRARY_EXPAND('+symbol+', ops)')
    (folder/'csrc/disentangled_attention_v4.cu').write_text(cu)
    (folder/'csrc/torch_binding.cpp').write_text(binding)
    (folder/'CMakeLists.txt').write_text((root/'CMakeLists.txt').read_text().replace('gliner_t4',symbol))
    logs=[]
    for cmd in (['cmake','-S',str(folder),'-B',str(folder/'build'),'-DCMAKE_BUILD_TYPE=Release',
                 '-DCMAKE_CUDA_ARCHITECTURES='+str(cap[0]*10+cap[1]),'-DCMAKE_PREFIX_PATH='+torch.utils.cmake_prefix_path,
                 '-DCMAKE_CUDA_FLAGS=--ptxas-options=-v'],
                ['cmake','--build',str(folder/'build'),'--parallel','2']):
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        logs.append(result.stdout)
        if result.returncode:raise RuntimeError(result.stdout)
    log='\n'.join(logs);(root/(label+'-build.txt')).write_text(log)
    torch.ops.load_library(str(folder/'torch-ext'/symbol/(symbol+'.so')))
    op_sets[label]=getattr(torch.ops,symbol)
    experiment['builds'][label]={'source_sha256':hashlib.sha256(cu.encode()).hexdigest(),
       'resources':[s for s in log.splitlines() if any(k in s for k in ('registers','spill','stack frame'))]}
    emit('EXACT_BUILD',{'path':label,**experiment['builds'][label]})

from gliner2.processing.word_splitter import WhitespaceTokenSplitter
split_original=WhitespaceTokenSplitter.__call__
eval_original=fused.eval
@functools.lru_cache(maxsize=1024)
def split_cached(text,lower):return tuple(split_original(WhitespaceTokenSplitter(),text,lower))
def split_call(self,text,lower=True):
    if type(self) is not WhitespaceTokenSplitter:return split_original(self,text,lower)
    return iter(split_cached(text,lower))
def select(path):
    prepared=path.endswith('_prep')
    attention_path=path[:-5] if prepared else path
    cls.forward_prepared,cls.forward_packed=originals if attention_path=='triton' else custom
    gliner_t4.ops=op_sets.get(attention_path,base_ops);gliner_t4.VERSION=4
    WhitespaceTokenSplitter.__call__=split_call if prepared else split_original
    fused.eval=types.MethodType(lambda self:self,fused) if prepared else eval_original
def byte_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

select('base')
toy=torch.tensor([tokens[:1024]],device='cuda')
captured_plan=[]
original_attention=gliner_t4.attention
def capture_plan(*args,**kwargs):
    if not captured_plan:captured_plan.append((args,kwargs))
    return original_attention(*args,**kwargs)
gliner_t4.attention=capture_plan
try:forward(fused,toy,torch.ones_like(toy))
finally:gliner_t4.attention=original_attention
args,kwargs=captured_plan[0]
plan=types.SimpleNamespace(pos_key=args[3],pos_query=args[4],delta_to_local=args[5],position_offset=args[6])
radius=kwargs['radius']
strict_ok=set(op_sets)
failures={k:[] for k in op_sets}
tested=0
def check_raw(q,k,v,mask=None,lengths=None,cu=None,max_length=0,description=''):
    global tested
    heads=q.shape[0] if cu is not None else q.shape[1]
    args=(q,k,v,plan.pos_key[:heads].contiguous(),plan.pos_query[:heads].contiguous(),
          plan.delta_to_local,plan.position_offset,mask,gliner_t4.SCALE_LOG2,4,lengths,radius,cu,max_length)
    expected=base_ops.disentangled_attention(*args)
    assert byte_equal(expected,base_ops.disentangled_attention(*args)),'Original attention is nondeterministic'
    for path in list(strict_ok):
        actual=op_sets[path].disentangled_attention(*args)
        if not byte_equal(expected,actual):
            strict_ok.discard(path);failures[path].append(description)
    tested+=1
    return args

@torch.inference_mode()
def raw_checks():
    saved=None
    for seed in (17,89,271):
        torch.manual_seed(seed)
        for length in (1,7,31,32,33,63,64,65,96,127,128,129,200,255,256,257,384,439,512,513,755,1024):
            q,k,v=[torch.randn((2,2,length,64),device='cuda',dtype=torch.float16)*(.3 if seed==17 else 1 if seed==89 else 3) for _ in range(3)]
            for mode in ('none','right','holes','empty'):
                mask=None;lengths=None
                if mode!='none':
                    mask=torch.ones((2,length),device='cuda',dtype=torch.uint8)
                    if mode=='right':mask[1,max(1,length//3):]=0
                    elif mode=='holes':mask[:,1::3]=0
                    else:mask.zero_()
                    lengths=gliner_t4.real_lengths(mask)
                args=check_raw(q,k,v,mask,lengths,description=f'{seed}/{length}/{mode}')
                if seed==17 and length==65 and mode=='holes':saved=args
        for lens in ([1,33,65],[7,127,257],[32,200,513],[96,384,755],[1,1024]):
            total=sum(lens)
            q,k,v=[torch.randn((total,2,64),device='cuda',dtype=torch.float16).permute(1,0,2) for _ in range(3)]
            cu=torch.tensor([0]+list(__import__('itertools').accumulate(lens)),device='cuda',dtype=torch.int32)
            check_raw(q,k,v,cu=cu,max_length=max(lens),description=f'packed/{seed}/{lens}')
    return saved
saved_args=raw_checks()
experiment['raw_checks']={'cases':tested,'eligible':sorted(strict_ok),'failures':failures}
emit('RAW_EQUALITY',experiment['raw_checks'])
torch.save(tuple(t.cpu() if isinstance(t,torch.Tensor) else t for t in saved_args),root/'sanitizer-inputs.pt')

for length in (96,200,256,384,439,512,755,1024):
    for padded in (False,True):
        ids=torch.tensor([tokens[i*length:(i+1)*length] for i in range(16)],device='cuda')
        mask=torch.ones_like(ids)
        if padded:
            for i in range(16):mask[i,max(1,length-i*length//20):]=0
        select('base');expected=forward(fused,ids,mask).clone()
        row={'length':length,'padded':padded,'byte_equal':{},'ms':{},'samples':{}}
        for path in list(strict_ok):
            select(path);actual=forward(fused,ids,mask)
            equal=byte_equal(expected,actual);row['byte_equal'][path]=bool(equal)
            if not equal:strict_ok.discard(path)
        for path in ('triton','base',*sorted(strict_ok-{'base'})):
            select(path)
            for _ in range(2):forward(fused,ids,mask)
        paths=['triton','base',*sorted(strict_ok-{'base'})]
        values={k:[] for k in paths}
        for trial in range(5):
            order=paths[trial%len(paths):]+paths[:trial%len(paths)]
            for path in order:values[path].append(timed_encoder(ids,mask,path,repetitions=3))
        row['samples']=values;row['ms']={k:statistics.median(v) for k,v in values.items()}
        experiment['encoder'].append(row);emit('EXACT_ENCODER',row)

paths=['triton','triton_prep','base','base_prep',*sorted(strict_ok-{'base'})]
for path in paths:
    split_cached.cache_clear();extract(path,chunks)
full_outputs={};timings={path:[] for path in paths}
for trial in range(3):
    order=paths[trial%len(paths):]+paths[:trial%len(paths)]
    for path in order:
        split_cached.cache_clear()
        elapsed,outputs=extract(path,chunks)
        timings[path].append(elapsed);full_outputs[path]=outputs
        emit('EXACT_EXTRACTION',{'trial':trial,'path':path,'seconds':elapsed,'chunks_per_s':len(chunks)/elapsed})
for path,values in timings.items():
    experiment['extraction'][path]={'seconds':values,'chunks_per_s':len(chunks)/statistics.median(values)}
    reference='triton' if path.startswith('triton') else 'base'
    experiment['parity'][path]={'reference':reference,'complete_results_equal':full_outputs[path]==full_outputs[reference]}
    (root/(path+'-exact-outputs.json')).write_text(json.dumps(full_outputs[path]))

extra=[
 'अनन्या शर्मा दिल्ली में कावेरी एनालिटिक्स में काम करती हैं। उनका पता 42 अशोक मार्ग है।',
 'రవి కుమార్ హైదరాబాద్‌లో పనిచేస్తున్నారు. సీత రెడ్డి విజయవాడలో నివసిస్తున్నారు.',
 'ಕಿರಣ್ ಕುಮಾರ್ ಬೆಂಗಳೂರಿನಲ್ಲಿ ವಾಸಿಸುತ್ತಾರೆ. ಆಶಾ ರಾವ್ ಮೈಸೂರಿನಲ್ಲಿ ಕೆಲಸ ಮಾಡುತ್ತಾರೆ.',
 'ஆனந்த் குமார் சென்னையில் வசிக்கிறார். மீனா ரவி மதுரையில் வேலை செய்கிறார்.',
 'അനിൽ കുമാർ കൊച്ചിയിൽ താമസിക്കുന്നു. മീര നായർ തിരുവനന്തപുരത്ത് ജോലി ചെയ്യുന്നു.',
 'অরিন্দম দাস কলকাতায় থাকেন। মিতা সেন ঢাকায় কাজ করেন।',
 'علي حسن يعمل في القاهرة. تسكن مريم أحمد في دبي في شارع النور.',
 'علی رضا لاہور میں رہتے ہیں۔ عائشہ خان کراچی میں کام کرتی ہیں۔',
 '山田太郎は東京に住んでいます。佐藤花子は大阪で働いています。',
 '김민수는 서울에 살고 있습니다. 이서연은 부산에서 일합니다.',
 '张伟在北京工作。李娜住在上海。',
 'Ольга Иванова живёт в Москве. Алексей Смирнов работает в Казани.',
 'Elena López trabaja en Madrid para Cedarline Services. Vive en Calle Mayor 42.',
 'Müller arbeitet in Berlin. Anna Schmidt wohnt in der Hauptstraße 17.',
 'İpek Yılmaz Ankara’da çalışıyor. Ahmet Kaya İstanbul’da yaşıyor.',
 'Jean Dupont travaille à Paris. Marie Noël habite au 42 rue des Écoles.',
 'Elara Vossmere at Cedarline Services in Tokyo, 220 Midori Terrace, Apartment 18. '*30,
 'A\tB\nNames: Elara Vossmere\nCity: São Paulo\nStreet: Rua das Acácias\n',
 '', '  \n\t ', 'Elara Vossmere '*200]
extra_outputs={}
for path in paths:
    split_cached.cache_clear();_,extra_outputs[path]=extract(path,extra)
    reference='triton' if path.startswith('triton') else 'base'
    if path in ('triton','base'):continue
    experiment['multilingual_parity'][path]={'cases':len(extra),'reference':reference,
        'complete_results_equal':extra_outputs[path]==extra_outputs[reference]}
    (root/(path+'-extra-outputs.json')).write_text(json.dumps(extra_outputs[path]))

experiment['accepted_exact_candidates']=[path for path in paths if path not in ('triton','base')
    and experiment['parity'][path]['complete_results_equal'] and experiment['multilingual_parity'][path]['complete_results_equal']]
experiment['native_vs_custom_complete_results_equal']=full_outputs['triton']==full_outputs['base']
experiment['no_fast_count']=True
(root/'exact-result.json').write_text(json.dumps(experiment,indent=2))
with zipfile.ZipFile(root/'exact-export.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in root.iterdir():
        if p.name.endswith(('-exact-outputs.json','-extra-outputs.json','-build.txt')) or p.name=='exact-result.json':z.write(p,p.name)
    for label in variant_sources:
        z.write(root/label/'csrc/disentangled_attention_v4.cu',label+'/disentangled_attention_v4.cu')
emit('EXACT_RESULT',experiment)
