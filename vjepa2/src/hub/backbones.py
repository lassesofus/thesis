# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

# for testing
# VJEPA_BASE_URL = "http://localhost:8300"

ARCH_NAME_MAP = {
    "vit_large": ("vit_large", "vitl"),
    "vit_huge": ("vit_huge", "vith"),
    "vit_giant": ("vit_giant_xformers", "vitg"),
    "vit_ac_giant": ("vit_giant_xformers", "vjepa2-ac-vitg"),
    "vit_giant_384": ("vit_giant_xformers", "vitg-384"),
}


def _clean_backbone_key(state_dict):
    for key, val in state_dict.copy().items():
        _ = state_dict.pop(key)
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        state_dict[key] = val
    return state_dict


def _make_vjepa2_ac_model(
    *,
    model_name: str = "vit_ac_giant",
    img_size=256,
    patch_size=16,
    tubelet_size=2,
    num_frames=64,
    pretrained: bool = True,
    **kwargs,
):
    from ..models import ac_predictor as vit_ac_predictor
    from ..models import vision_transformer as vit_encoder

    vit_encoder_kwargs = dict(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
    )
    vit_encoder_kwargs.update(**kwargs)

    arch_name = ARCH_NAME_MAP[model_name][0]
    encoder = vit_encoder.__dict__[arch_name](**vit_encoder_kwargs)

    vit_predictor_kwargs = dict(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.embed_dim,
    )
    vit_predictor_kwargs.update(**kwargs)

    predictor = vit_ac_predictor.__dict__["vit_ac_predictor"](**vit_predictor_kwargs)

    if pretrained:
        model_file = ARCH_NAME_MAP[model_name][-1]
        url = VJEPA_BASE_URL + f"/{model_file}.pt"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        encoder_state_dict = _clean_backbone_key(state_dict["encoder"])
        encoder.load_state_dict(encoder_state_dict, strict=False)
        predictor_state_dict = _clean_backbone_key(state_dict["predictor"])
        predictor.load_state_dict(predictor_state_dict, strict=True)

    return encoder, predictor


def _make_vjepa2_model(
    *,
    model_name: str = "vit_large",
    img_size=256,
    patch_size=16,
    tubelet_size=2,
    num_frames=64,
    pretrained: bool = True,
    **kwargs,
):
    from ..models import predictor as vit_predictor
    from ..models import vision_transformer as vit_encoder

    vit_encoder_kwargs = dict(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
    )
    vit_encoder_kwargs.update(**kwargs)

    arch_name = ARCH_NAME_MAP[model_name][0]
    encoder = vit_encoder.__dict__[arch_name](**vit_encoder_kwargs)

    vit_predictor_kwargs = dict(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        use_mask_tokens=True,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=384,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        depth=12,
        num_heads=12,
        num_mask_tokens=10,
        use_rope=True,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
    )
    vit_predictor_kwargs.update(**kwargs)

    predictor = vit_predictor.__dict__["vit_predictor"](**vit_predictor_kwargs)

    if pretrained:
        model_file = ARCH_NAME_MAP[model_name][-1]
        url = VJEPA_BASE_URL + f"/{model_file}.pt"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        encoder_state_dict = _clean_backbone_key(state_dict["encoder"])
        encoder.load_state_dict(encoder_state_dict, strict=False)  # state_dict has pos_embed but we use RoPE
        predictor_state_dict = _clean_backbone_key(state_dict["predictor"])
        predictor.load_state_dict(predictor_state_dict, strict=False)  # state_dict has pos_embed but we use RoPE

    return encoder, predictor


def vjepa2_vit_large(*, pretrained: bool = True, **kwargs):
    """
    VJEPA 2 ViT-Large model
    """
    return _make_vjepa2_model(model_name="vit_large", img_size=256, pretrained=pretrained, **kwargs)


def vjepa2_vit_huge(*, pretrained: bool = True, **kwargs):
    """
    VJEPA 2 ViT-Huge model
    """
    return _make_vjepa2_model(model_name="vit_huge", img_size=256, pretrained=pretrained, **kwargs)


def vjepa2_vit_giant(*, pretrained: bool = True, **kwargs):
    """
    VJEPA 2 ViT-giant model
    """
    return _make_vjepa2_model(model_name="vit_giant", img_size=256, pretrained=pretrained, **kwargs)


def vjepa2_vit_giant_384(*, pretrained: bool = True, **kwargs):
    """
    VJEPA 2 ViT-giant-384 model
    """
    return _make_vjepa2_model(model_name="vit_giant_384", img_size=384, pretrained=pretrained, **kwargs)


def vjepa2_ac_vit_giant(*, pretrained: bool = True, **kwargs):
    """
    VJEPA 2-AC ViT-giant model
    """
    return _make_vjepa2_ac_model(model_name="vit_ac_giant", img_size=256, pretrained=pretrained, **kwargs)
