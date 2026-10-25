# A small random-weight gliner2 2.1.0 span model with a locally trained tokenizer: no Hub download, CPU friendly.
# It runs the same gliner2 collation, encoder, count head, span scoring and decoding code as the real checkpoint, so
# code-path changes can be compared output for output; its scores mean nothing.
import random

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
from transformers import PreTrainedTokenizerFast, DebertaV2Config
from gliner2.inference.engine import GLiNER2 as Extractor
from gliner2.configuration import ExtractorConfig

WORDS = ("john smith anna mueller acme corp gmbh berlin munich main street road avenue invoice policy number "
         "the a of and to in for with on at by from paid delivered contract signed meeting office").split()

def corpus(n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(20, 500)
        out.append(" ".join(rng.choice(WORDS).capitalize() if rng.random() < .2 else rng.choice(WORDS) for _ in range(k)) + ".")
    return out

def build(seed=0):
    torch.manual_seed(seed)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(corpus(200) + ["Person Organization Street City Insurance Number description NOT names"],
                            trainers.WordPieceTrainer(vocab_size=400, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    enc = DebertaV2Config(vocab_size=len(tokenizer), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                          intermediate_size=128, relative_attention=True, position_buckets=256,
                          pos_att_type=["p2c", "c2p"], max_relative_positions=-1, norm_rel_ebd="layer_norm",
                          share_att_key=True, position_biased_input=False, max_position_embeddings=512, type_vocab_size=0)
    cfg = ExtractorConfig(model_name="standin", max_width=8, counting_layer="count_lstm", attn_implementation="eager")
    model = Extractor(cfg, encoder_config=enc, tokenizer=tokenizer)
    # random init gives near-zero scores; widen them so thresholds keep real spans in the comparison
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(1.5)
    return model.eval()
