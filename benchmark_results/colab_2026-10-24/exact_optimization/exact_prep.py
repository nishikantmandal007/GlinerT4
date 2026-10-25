"""Exact token-ID lookup for a frozen inference tokenizer; no model math changes."""
import types


def install_bulk_token_id_lookup(tokenizer):
    """Install only after exhaustively checking the current fast-tokenizer vocabulary."""
    backend=getattr(tokenizer,'_tokenizer',None)
    if not getattr(tokenizer,'is_fast',False) or backend is None:
        return False
    original=tokenizer.convert_tokens_to_ids
    original_function=getattr(original,'__func__',None)
    if original_function is None:
        return False
    vocabulary=tokenizer.get_vocab()
    size=len(tokenizer)
    if len(vocabulary)!=size:
        return False
    tokens=list(vocabulary)
    if original(tokens)!=[vocabulary[token] for token in tokens]:
        return False

    def convert(self,tokens):
        # shortcut: vocabulary is immutable during inference; reinstall after replacing it in place.
        if (type(tokens) is list and self._tokenizer is backend and len(self)==size
                and all(type(token) is str for token in tokens)):
            ids=[vocabulary.get(token) for token in tokens]
            if None not in ids:
                return ids
        return original_function(self,tokens)

    tokenizer.convert_tokens_to_ids=types.MethodType(convert,tokenizer)
    return True
