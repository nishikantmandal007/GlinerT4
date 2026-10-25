import ast, functools, types, cProfile, pstats, io
from pathlib import Path
root=Path('/content/gliner-ab')
source=(root/'benchmark.py').read_text()
setup=ast.parse(source[:source.index('for length in (256, 384, 512, 1024):')])
setup.body=[n for n in setup.body if not (
 isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='plain' for t in n.targets)
 or isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
 and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='plain')]
exec(compile(setup,'setup','exec'))
exec(compile(source[source.index('labels = {'):source.index("for path in ('triton','custom'):\n    extract(path, chunks)")],'helpers','exec'))
assert torch.cuda.device_count()==1 and cap==(7,5)
from gliner2.processing.word_splitter import WhitespaceTokenSplitter
sys.path.insert(0,str(root))
from exact_prep import install_bulk_token_id_lookup
split_original=WhitespaceTokenSplitter.__call__
@functools.lru_cache(maxsize=1024)
def split_cached(text,lower):return tuple(split_original(WhitespaceTokenSplitter(),text,lower))
def split_call(self,text,lower=True):
 if type(self) is not WhitespaceTokenSplitter:return split_original(self,text,lower)
 return iter(split_cached(text,lower))
select('custom')
extract('custom',chunks)
reference_seconds,reference_outputs=extract('custom',chunks)
(root/'baseline-outputs.json').write_text(json.dumps(reference_outputs))
WhitespaceTokenSplitter.__call__=split_call
fused.eval=types.MethodType(lambda self:self,fused)
lookup_installed=install_bulk_token_id_lookup(tokenizer)
baseline_samples=[]
for trial in range(3):
 split_cached.cache_clear();elapsed,out=extract('custom',chunks)
 assert out==reference_outputs,'Preparation changed full results'
 baseline_samples.append(elapsed);emit('T4_BASELINE',{'trial':trial,'seconds':elapsed,'chunks_per_s':len(chunks)/elapsed})
prof=cProfile.Profile();split_cached.cache_clear();prof.enable();extract('custom',chunks[:97]);prof.disable()
s=io.StringIO();pstats.Stats(prof,stream=s).sort_stats('cumulative').print_stats(40)
(root/'cpu-profile.txt').write_text(s.getvalue());print(s.getvalue(),flush=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
 extract('custom',chunks[:97])
p.export_chrome_trace(str(root/'t4-trace.json'))
print(p.key_averages().table(sort_by='self_cuda_time_total',row_limit=25),flush=True)
(root/'gpu-profile.txt').write_text(p.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
report.update({'reference_seconds':reference_seconds,'baseline_samples':baseline_samples,'baseline_chunks_per_s':len(chunks)/statistics.median(baseline_samples),'bulk_lookup_installed':lookup_installed,'prep_full_json_equal':True})
(root/'t4-baseline.json').write_text(json.dumps(report,indent=2))
# Capture real shape plan after complete model preparation for direct attention gates.
select('custom');captured=[];orig_attention=gliner_t4.attention

def capture(*args,**kwargs):
 if not captured:captured.append((args,kwargs))
 return orig_attention(*args,**kwargs)
gliner_t4.attention=capture
try:
 toy=torch.tensor([tokens[:1024]],device='cuda');forward(fused,toy,torch.ones_like(toy))
finally:gliner_t4.attention=orig_attention
args,kwargs=captured[0]
plan=types.SimpleNamespace(pos_key=args[3],pos_query=args[4],delta_to_local=args[5],position_offset=args[6]);radius=kwargs['radius']
base_ops=gliner_t4.ops
print('T4_BASELINE_DONE',flush=True)
