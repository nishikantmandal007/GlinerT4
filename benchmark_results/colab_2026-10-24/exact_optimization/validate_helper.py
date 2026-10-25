import copy,json
from pathlib import Path
namespace={}
exec('"""Exact token-ID lookup for a frozen inference tokenizer; no model math changes."""\nimport types\n\n\ndef install_bulk_token_id_lookup(tokenizer):\n    """Install only after exhaustively checking the current fast-tokenizer vocabulary."""\n    backend=getattr(tokenizer,\'_tokenizer\',None)\n    if not getattr(tokenizer,\'is_fast\',False) or backend is None:\n        return False\n    original=tokenizer.convert_tokens_to_ids\n    original_function=getattr(original,\'__func__\',None)\n    if original_function is None:\n        return False\n    vocabulary=tokenizer.get_vocab()\n    size=len(tokenizer)\n    if len(vocabulary)!=size:\n        return False\n    tokens=list(vocabulary)\n    if original(tokens)!=[vocabulary[token] for token in tokens]:\n        return False\n\n    def convert(self,tokens):\n        # shortcut: vocabulary is immutable during inference; reinstall after replacing it in place.\n        if (type(tokens) is list and self._tokenizer is backend and len(self)==size\n                and all(type(token) is str for token in tokens)):\n            ids=[vocabulary.get(token) for token in tokens]\n            if None not in ids:\n                return ids\n        return original_function(self,tokens)\n\n    tokenizer.convert_tokens_to_ids=types.MethodType(convert,tokenizer)\n    return True\n',namespace)
install=namespace['install_bulk_token_id_lookup']
# Exercise the actual tokenizer without mutating the serving model's vocabulary.
tokenizer.convert_tokens_to_ids=original_convert
clone=copy.deepcopy(tokenizer)
original_clone=clone.convert_tokens_to_ids
assert install(clone)
entries=list(clone.get_vocab())
assert clone.convert_tokens_to_ids(tokens=entries)==original_clone(entries)
for value in special:
 assert outcome(clone.convert_tokens_to_ids,value)==outcome(original_clone,value)
new_token='[exact-lookup-new-token-for-fallback]'
clone.add_tokens([new_token])
assert clone.convert_tokens_to_ids([new_token])==original_clone([new_token])
clone_after=copy.deepcopy(clone)
clone_after.add_tokens(['[second-new-token-after-copy]'])
assert clone_after.convert_tokens_to_ids(['[second-new-token-after-copy]'])==type(clone_after).convert_tokens_to_ids(clone_after,['[second-new-token-after-copy]'])
# Check the exported helper itself through the real extraction API.
saved_select('triton_prep')
tokenizer.convert_tokens_to_ids=original_convert
assert install(tokenizer)
with torch.inference_mode():
 actual=[fused.batch_extract_entities(chunks,p,batch_size=16,threshold=.65,include_confidence=True,include_spans=True) for p in passes]
assert actual==full_outputs['triton']
with torch.inference_mode():
 actual_extra=[fused.batch_extract_entities(extra,p,batch_size=16,threshold=.65,include_confidence=True,include_spans=True) for p in passes]
assert actual_extra==extra_outputs['triton']
Path('/content/gliner-ab/helper-check.json').write_text(json.dumps({'all_vocabulary_entries':len(entries),'fallback_cases':len(special),'vocabulary_growth_fallback':True,'copied_tokenizer_fallback':True,'keyword_argument':True,'full_triton_extraction_equal':True,'extra_cases_equal':len(extra)}))
print('EXPORTED_HELPER_CHECK_PASSED',len(entries),flush=True)
