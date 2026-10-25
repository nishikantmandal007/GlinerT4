import random

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import DebertaV2Config, PreTrainedTokenizerFast

WORDS = ("john smith anna mueller acme corp gmbh berlin munich main street road avenue invoice policy number "
         "the a of and to in for with on at by from paid delivered contract signed meeting office").split()


def corpus(n, seed=0):
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        words = (rng.choice(WORDS).capitalize() if rng.random() < .2 else rng.choice(WORDS)
                 for _ in range(rng.randint(20, 500)))
        texts.append(" ".join(words) + ".")
    return texts


def build(seed=0):
    from gliner2.configuration import ExtractorConfig
    from gliner2.inference.engine import GLiNER2

    torch.manual_seed(seed)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(corpus(200) + ["Person Organization Street City description NOT names"],
                            trainers.WordPieceTrainer(vocab_size=400,
                                                      special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                        sep_token="[SEP]", mask_token="[MASK]")
    encoder = DebertaV2Config(vocab_size=len(tokenizer), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                              intermediate_size=128, relative_attention=True, position_buckets=256,
                              pos_att_type=["p2c", "c2p"], max_relative_positions=-1, norm_rel_ebd="layer_norm",
                              share_att_key=True, position_biased_input=False, max_position_embeddings=512,
                              type_vocab_size=0)
    config = ExtractorConfig(model_name="standin", max_width=8, counting_layer="count_lstm",
                             attn_implementation="eager")
    model = GLiNER2(config, encoder_config=encoder, tokenizer=tokenizer)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(1.5)
    return model.eval()
