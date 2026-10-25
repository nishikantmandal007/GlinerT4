import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_add_layernorm_is_bit_identical():
    from glinert4.ops import ops

    torch.manual_seed(0)
    weight = torch.randn(768, device="cuda", dtype=torch.float16)
    bias = torch.randn(768, device="cuda", dtype=torch.float16)
    for rows in (1, 7, 4096):
        x = torch.randn(rows, 768, device="cuda", dtype=torch.float16)
        residual = torch.randn(rows, 768, device="cuda", dtype=torch.float16)
        expected = torch.nn.functional.layer_norm(x + residual, (768,), weight, bias, 1e-7)
        assert torch.equal(ops().add_layernorm(x, residual, weight, bias, 1e-7), expected)


@pytest.mark.parametrize("length", [96, 384, 1024])
@pytest.mark.parametrize("padded", [False, True])
def test_encoder_matches_disentangled_flash(length, padded):
    from gliner2 import AutoExtractor

    from glinert4 import attention

    model = AutoExtractor.from_pretrained("fastino/gliner2-multi-v1", map_location="cuda", quantize=True,
                                          attention_backend="disentangled_flash").eval()
    torch.manual_seed(0)
    ids = torch.randint(1000, 30000, (16, length), device="cuda")
    mask = torch.ones_like(ids)
    if padded:
        for row in range(16):
            mask[row, max(1, length - row * length // 20):] = 0
    with torch.inference_mode():
        attention.uninstall()
        reference = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        attention.install()
        ours = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state.float()
    keep = mask.bool()[..., None]
    delta = (ours - reference).abs().masked_select(keep)
    assert torch.isfinite(ours).all()
    assert delta.mean().item() < 5e-3
