#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the WebBrain DeepSeek-V4-Flash-Vision-BF16 vision overlay into a llama.cpp
mmproj GGUF (KIMIK25 projector type) that can be loaded with --mmproj.

Inputs (two safetensors, both BF16):
  vision_tower.safetensors   - MoonViT-3d tower (Kimi-K2.6), 329 tensors
  mm_projector.safetensors   - WebBrain PatchMerger projector, 6 tensors

Output:
  mmproj-v4-flash-vision-bf16.gguf

The output is a "clip" architecture GGUF with clip.projector_type == "kimik25",
matching the build_kimik25() graph in examples/mtmd/clip.cpp. It does NOT contain
the DeepSeek text backbone; it is a standalone multimodal projector meant to be
used together with a DeepSeek V4 Flash text GGUF via --mmproj. It also embeds the
64-ID hash-layer expert routing palette as clip.vision.routing_palette metadata,
which the mtmd runtime feeds through the text backbone's tid2eid lookup for image
tokens (which otherwise carry no token ids).

Usage:
  python tools/convert_v4flash_vision_mmproj.py \
      --vision-tower R:/models/vision_tower.safetensors \
      --mm-projector R:/models/mm_projector.safetensors \
      --outfile R:/models/mmproj-v4-flash-vision-bf16.gguf
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# allow importing the local gguf-py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))

import gguf  # noqa: E402


# ---------------------------------------------------------------------------
# safetensors reader (no torch / safetensors dependency needed)
# ---------------------------------------------------------------------------
class SafeTensorFile:
    """Minimal safetensors reader. Only reads the tensors we ask for."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_len)
            self.data_offset = 8 + header_len
        self.header = json_loads(header_bytes)
        self.offsets = {}
        for name, info in self.header.items():
            if name == "__metadata__":
                continue
            self.offsets[name] = (info["data_offsets"][0], info["data_offsets"][1])

    def get(self, name: str) -> np.ndarray:
        dtype_map = {
            "BF16": np.dtype("<u2"),
            "F16": np.dtype("<f2"),
            "F32": np.dtype("<f4"),
            "F64": np.dtype("<f8"),
            "I8": np.dtype("<i1"),
            "I16": np.dtype("<i2"),
            "I32": np.dtype("<i4"),
            "I64": np.dtype("<i8"),
            "U8": np.dtype("u1"),
            "U16": np.dtype("<u2"),
            "U32": np.dtype("<u4"),
            "U64": np.dtype("<u8"),
        }
        info = self.header[name]
        dtype = dtype_map[info["dtype"]]
        shape = list(info["shape"])
        start, end = self.offsets[name]
        with open(self.path, "rb") as f:
            f.seek(self.data_offset + start)
            raw = f.read(end - start)
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
        # keep BF16 as raw uint16 (2 bytes) so GGUF stores it as BF16, not float32
        return arr

    def keys(self):
        return [k for k in self.header if k != "__metadata__"]


def json_loads(data: bytes) -> dict:
    import json
    # safetensors header is JSON with padding to multiple of 8 bytes
    # strip trailing whitespace padding
    return json.loads(data.rstrip(b" \t\r\n\x00"))


def bf16_to_f32(arr: np.ndarray) -> np.ndarray:
    """Convert a uint16 BF16 array to float32 (in-place safe)."""
    u32 = arr.astype(np.uint32, copy=False).view(np.uint32) << 16
    return u32.view(np.float32)


# ---------------------------------------------------------------------------
# Kimi-K2.5 / K2.6 QKV permute (interleaved -> split RoPE), matches upstream
# KimiK25Model.permute()
# ---------------------------------------------------------------------------
def permute_qk(weights: np.ndarray, n_head: int) -> np.ndarray:
    out_dim, in_dim = weights.shape
    head_dim = out_dim // n_head
    # reshape(n_head, head_dim//4, 2, 2, in_dim) then permute(0,2,1,3,4)
    w = weights.reshape(n_head, head_dim // 4, 2, 2, in_dim)
    w = w.transpose(0, 2, 1, 3, 4)
    return w.reshape(out_dim, in_dim)


def permute_qk_bias(bias: np.ndarray, n_head: int) -> np.ndarray:
    head_dim = bias.shape[0] // n_head
    b = bias.reshape(n_head, head_dim // 4, 2, 2)
    b = b.transpose(0, 2, 1, 3)
    return b.reshape(-1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Convert WebBrain V4-Flash vision overlay to mmproj GGUF")
    ap.add_argument("--vision-tower", required=True, help="path to vision_tower.safetensors")
    ap.add_argument("--mm-projector", required=True, help="path to mm_projector.safetensors")
    ap.add_argument("--outfile", required=True, help="output .gguf path")
    args = ap.parse_args()

    vt_path = Path(args.vision_tower)
    mp_path = Path(args.mm_projector)

    # hyperparameters (from Kimi-K2.6 vision_config / preprocessor_config)
    n_embd = 1152
    n_ff = 4304
    n_head = 16
    n_layer = 27
    patch_size = 14
    image_size = 64 * patch_size  # 896
    scale_factor = 2
    eps = 1e-5
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    # merged token area (patch 2x2 merged by the PatchMerger)
    patch_area = patch_size * patch_size * scale_factor * scale_factor  # 784
    min_pixels = 8 * patch_size * patch_size
    # cap merged image tokens at 512 to match deepseek_vision.max_image_tokens
    # in the official config.json: max_pixels = 512 * patch_area = 401408
    max_pixels = 512 * patch_area

    # DeepSeek V4 Flash hash-layer routing palette (64 expert token ids).
    # The text backbone's hash layers select experts deterministically from token ids, but
    # image embeddings carry no token ids. This fixed 64-ID palette (cycled by image-local
    # offset) is embedded in the mmproj metadata so the mtmd runtime can feed these ids
    # through the hash-layer tid2eid lookup instead of the untrained gate fallback.
    # Source: configs/routing/deepseek-v4-flash-60d8d707-palette64.json
    routing_palette = [
        0, 1, 2, 8, 9, 10, 12, 74, 81, 110, 114, 240, 17081, 25312, 30711, 58279,
        7637, 8936, 45556, 52073, 7743, 8347, 13203, 19795, 44418, 62970, 79038, 6381, 48025, 109859, 29629, 91213,
        90662, 121562, 8570, 25568, 3685, 81916, 14638, 50590, 101211, 24832, 75337, 131, 15170, 79723, 84052, 20866,
        48327, 72234, 15507, 128, 16760, 34135, 36264, 59037, 3839, 29854, 109646, 64, 23442, 6584, 10255, 17173,
    ]

    print(f"reading vision tower from {vt_path}")
    vt = SafeTensorFile(vt_path)
    print(f"  {len(vt.keys())} tensors")

    print(f"reading mm projector from {mp_path}")
    mp = SafeTensorFile(mp_path)
    print(f"  {len(mp.keys())} tensors")

    writer = gguf.GGUFWriter(path=str(args.outfile), arch="clip", use_temp_file=False)

    # --- metadata ----------------------------------------------------------
    writer.add_name("DeepSeek-V4-Flash-Vision")
    writer.add_description("WebBrain MoonViT (Kimi-K2.6) + PatchMerger projector, KIMIK25 mmproj for DeepSeek V4 Flash")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    writer.add_bool("clip.has_text_encoder", False)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.has_llava_projector", True)
    writer.add_string("clip.projector_type", "kimik25")
    writer.add_bool("clip.use_gelu", True)

    writer.add_uint32("clip.vision.image_size", image_size)
    writer.add_uint32("clip.vision.patch_size", patch_size)
    # projection_dim must match the text model's embedding_length (required by clip loader)
    writer.add_uint32("clip.vision.projection_dim", 4096)
    writer.add_uint32("clip.vision.embedding_length", n_embd)
    writer.add_uint32("clip.vision.feed_forward_length", n_ff)
    writer.add_uint32("clip.vision.block_count", n_layer)
    writer.add_uint32("clip.vision.attention.head_count", n_head)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", eps)
    writer.add_uint32("clip.vision.projector.scale_factor", scale_factor)
    writer.add_uint32("clip.vision.image_min_pixels", min_pixels)
    writer.add_uint32("clip.vision.image_max_pixels", max_pixels)
    writer.add_array("clip.vision.image_mean", image_mean)
    writer.add_array("clip.vision.image_std", image_std)
    writer.add_array("clip.vision.routing_palette", routing_palette)

    # --- vision tower tensors ---------------------------------------------
    def add_tensor(name: str, arr: np.ndarray, force_f32: bool = False):
        # 1D tensors (biases, norm weights) and position embeddings must be F32:
        # upstream forces n_dims <= 1 (and _norm.weight) to F32, and the CUDA
        # bin_bcast kernels only accept F32/F16 second operands.
        if force_f32 or arr.ndim <= 1:
            # arr is raw uint16 BF16 bits; convert to float32
            data = bf16_to_f32(arr)
            writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            # arr is raw uint16 BF16 bits; store as BF16 (2 bytes)
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.BF16)

    # patch embed (conv2d) and positional embedding
    add_tensor("v.patch_embd.weight", vt.get("patch_embed.proj.weight"))
    add_tensor("v.patch_embd.bias", vt.get("patch_embed.proj.bias"))
    pos_emb = vt.get("patch_embed.pos_emb.weight")  # [64, 64, 1152]
    # clip expects ne[0]=C, ne[1]=W, ne[2]=H; GGUF stores dims reversed from numpy
    # numpy [64,64,1152] -> GGUF ne [1152,64,64]. Keep as-is.
    add_tensor("v.position_embd.weight", pos_emb, force_f32=True)

    for il in range(n_layer):
        pre = f"encoder.blocks.{il}"

        # attention qkv (merged) with Q/K interleaved->split permute
        wqkv = vt.get(f"{pre}.wqkv.weight")  # [3456, 1152]
        qkv_dim = wqkv.shape[0] // 3
        wq = permute_qk(wqkv[:qkv_dim], n_head)
        wk = permute_qk(wqkv[qkv_dim:2 * qkv_dim], n_head)
        wv = wqkv[2 * qkv_dim:]
        add_tensor(f"v.blk.{il}.attn_qkv.weight", np.concatenate([wq, wk, wv], axis=0))

        bqkv = vt.get(f"{pre}.wqkv.bias")  # [3456]
        bq = permute_qk_bias(bqkv[:qkv_dim], n_head)
        bk = permute_qk_bias(bqkv[qkv_dim:2 * qkv_dim], n_head)
        bv = bqkv[2 * qkv_dim:]
        add_tensor(f"v.blk.{il}.attn_qkv.bias", np.concatenate([bq, bk, bv]))

        # output projection
        add_tensor(f"v.blk.{il}.attn_out.weight", vt.get(f"{pre}.wo.weight"))
        add_tensor(f"v.blk.{il}.attn_out.bias", vt.get(f"{pre}.wo.bias"))

        # layer norms
        add_tensor(f"v.blk.{il}.ln1.weight", vt.get(f"{pre}.norm0.weight"))
        add_tensor(f"v.blk.{il}.ln1.bias", vt.get(f"{pre}.norm0.bias"))
        add_tensor(f"v.blk.{il}.ln2.weight", vt.get(f"{pre}.norm1.weight"))
        add_tensor(f"v.blk.{il}.ln2.bias", vt.get(f"{pre}.norm1.bias"))

        # ffn: fc0 = up (input->n_ff), fc1 = down (n_ff->n_embd)
        add_tensor(f"v.blk.{il}.ffn_up.weight", vt.get(f"{pre}.mlp.fc0.weight"))
        add_tensor(f"v.blk.{il}.ffn_up.bias", vt.get(f"{pre}.mlp.fc0.bias"))
        add_tensor(f"v.blk.{il}.ffn_down.weight", vt.get(f"{pre}.mlp.fc1.weight"))
        add_tensor(f"v.blk.{il}.ffn_down.bias", vt.get(f"{pre}.mlp.fc1.bias"))

    # --- projector tensors -------------------------------------------------
    add_tensor("mm.input_norm.weight", mp.get("pre_norm.weight"))
    add_tensor("mm.input_norm.bias", mp.get("pre_norm.bias"))
    add_tensor("mm.1.weight", mp.get("proj.0.weight"))
    add_tensor("mm.1.bias", mp.get("proj.0.bias"))
    add_tensor("mm.2.weight", mp.get("proj.2.weight"))
    add_tensor("mm.2.bias", mp.get("proj.2.bias"))

    # --- write -------------------------------------------------------------
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"done. wrote {args.outfile}")


if __name__ == "__main__":
    main()