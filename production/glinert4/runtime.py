import functools
import types

import torch

GLINER2_VERSION = "2.1.0"

_word_cache = {}


def check_gliner2():
    import gliner2

    if gliner2.__version__ != GLINER2_VERSION:
        raise RuntimeError(f"glinert4 requires gliner2=={GLINER2_VERSION}, found {gliner2.__version__}")


def install_word_cache(size=1024):
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    if WhitespaceTokenSplitter in _word_cache:
        return
    stock = WhitespaceTokenSplitter.__call__

    @functools.lru_cache(maxsize=size)
    def words(text, lower):
        return tuple(stock(WhitespaceTokenSplitter(), text, lower))

    def cached(self, text, lower=True):
        if type(self) is not WhitespaceTokenSplitter:
            return stock(self, text, lower)
        return iter(words(text, lower))

    WhitespaceTokenSplitter.__call__ = cached
    _word_cache[WhitespaceTokenSplitter] = stock


def uninstall_word_cache():
    for cls, stock in _word_cache.items():
        cls.__call__ = stock
    _word_cache.clear()


def add_length_buckets(model, bucket):
    def pad(encoder, args, kwargs):
        input_ids, attention_mask = kwargs["input_ids"], kwargs["attention_mask"]
        encoder.unpadded_length = length = input_ids.shape[1]
        extra = -length % bucket
        if extra:
            kwargs["input_ids"] = torch.nn.functional.pad(input_ids, (0, extra), value=encoder.config.pad_token_id or 0)
            kwargs["attention_mask"] = torch.nn.functional.pad(attention_mask, (0, extra), value=0)
        return args, kwargs

    def cut(encoder, args, kwargs, output):
        output.last_hidden_state = output.last_hidden_state[:, :encoder.unpadded_length]
        return output

    model.encoder.register_forward_pre_hook(pad, with_kwargs=True, prepend=True)
    model.encoder.register_forward_hook(cut, with_kwargs=True)


def freeze_eval(model):
    model.eval()
    model.eval = types.MethodType(lambda self: self, model)
    return model


def load_model(checkpoint, device, length_bucket):
    from gliner2 import AutoExtractor

    on_gpu = torch.device(device).type == "cuda"
    options = {"attention_backend": "disentangled_flash"} if on_gpu else {}
    model = AutoExtractor.from_pretrained(checkpoint, map_location=device, quantize=on_gpu, **options)
    if on_gpu and length_bucket > 1:
        add_length_buckets(model, length_bucket)
    freeze_eval(model)
    if on_gpu:
        torch.cuda.empty_cache()
    return model
