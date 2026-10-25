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
