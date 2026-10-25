"""Replace per-token Python/Rust calls with one exact vocabulary-list lookup."""
import json
from pathlib import Path
import statistics
import types
import zipfile

root=Path('/content/gliner-ab')
tokenizer=fused.processor.tokenizer
original_convert=tokenizer.convert_tokens_to_ids
backend=tokenizer._tokenizer
vocabulary=tokenizer.get_vocab()
vocabulary_size=len(tokenizer)
assert len(vocabulary)==vocabulary_size

def bulk_convert(self,value):
    # shortcut: this inference tokenizer is immutable; reinstall after in-place vocabulary replacement.
    if (type(value) is list and self._tokenizer is backend and len(self)==vocabulary_size
            and all(type(token) is str for token in value)):
        ids=[vocabulary.get(token) for token in value]
        if None not in ids:return ids
    return original_convert(value)

patched_convert=types.MethodType(bulk_convert,tokenizer)
all_tokens=list(vocabulary)
assert original_convert(all_tokens)==patched_convert(all_tokens),'Vocabulary lookup mismatch'
def outcome(function,value):
    try:return ('result',function(value))
    except Exception as e:return ('error',type(e).__name__,str(e))
special=[None,[],['[not_a_real_vocab_item]'],['\ud800'],['known',7],['known',[]],('Delhi','東京'),'Delhi','İstanbul']
for value in special:
    assert outcome(original_convert,value)==outcome(patched_convert,value),'Fallback behavior changed'
saved_select=select
def select(path):
    bulk=path.endswith('_bulk')
    saved_select(path[:-5]+'_prep' if bulk else path)
    tokenizer.convert_tokens_to_ids=patched_convert if bulk else original_convert

paths=['triton_prep','triton_bulk','base_prep','base_bulk','async_simple_bulk','async_pipeline_bulk']
paths=[path for path in paths if not path.startswith('async_pipeline') or 'async_pipeline' in op_sets]
for path in paths:
    split_cached.cache_clear();extract(path,chunks)
samples={path:[] for path in paths}
results={}
for trial in range(5):
    order=paths[trial%len(paths):]+paths[:trial%len(paths)]
    for path in order:
        split_cached.cache_clear()
        seconds,result=extract(path,chunks)
        reference='triton' if path.startswith('triton') else 'base'
        assert result==full_outputs[reference],path+' changed complete results'
        samples[path].append(seconds);results[path]=result
        emit('BULK_SAMPLE',{'trial':trial,'path':path,'seconds':seconds,'chunks_per_s':len(chunks)/seconds})
bulk_result={'vocabulary_entries_checked':len(all_tokens),'fallback_cases_checked':len(special),
             'gate':'Complete JSON including scores/spans equals unchanged baseline on every timed round.',
             'paths':{},'extra_parity':{}}
for path,values in samples.items():
    bulk_result['paths'][path]={'seconds':values,'chunks_per_s':len(chunks)/statistics.median(values),
                              'complete_results_equal':True,'reference':'triton' if path.startswith('triton') else 'base'}
    (root/(path+'-bulk-outputs.json')).write_text(json.dumps(results[path]))
    split_cached.cache_clear();_,result=extract(path,extra)
    reference='triton' if path.startswith('triton') else 'base'
    assert result==extra_outputs[reference],path+' changed extra-case scores or spans'
    bulk_result['extra_parity'][path]={'cases':len(extra),'complete_results_equal':True}
(root/'bulk-result.json').write_text(json.dumps(bulk_result,indent=2))
with zipfile.ZipFile(root/'bulk-export.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in root.iterdir():
        if p.name.endswith('-bulk-outputs.json') or p.name=='bulk-result.json':z.write(p,p.name)
emit('BULK_RESULT',bulk_result)
