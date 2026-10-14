import sys
import os
import json
import struct
import argparse
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required.  pip install numpy")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required.  pip install Pillow")
    sys.exit(1)


GX_FMT = {
    0: 'I4', 1: 'I8', 2: 'IA4', 3: 'IA8',
    4: 'RGB565', 5: 'RGB5A3', 6: 'RGBA8',
    8: 'C4', 9: 'C8', 10: 'C14X2', 14: 'CMPR',
}
TLUT_FMT = {0: 'IA8', 1: 'RGB565', 2: 'RGB5A3'}

GX_BLOCK = {
    0: (8, 8, 4), 1: (8, 4, 8), 2: (8, 4, 8), 3: (4, 4, 16),
    4: (4, 4, 16), 5: (4, 4, 16), 6: (4, 4, 32),
    8: (8, 8, 4), 9: (8, 4, 8), 10: (4, 4, 16), 14: (8, 8, 4),
}
INDEXED = {8: 16, 9: 256, 10: 16384}


def level_bytes(fmt, w, h):
    bw, bh, bpp = GX_BLOCK[fmt]
    return (-(-w // bw) * bw) * (-(-h // bh) * bh) * bpp // 8


def mip_chain_bytes(fmt, w, h, levels):
    return sum(level_bytes(fmt, max(1, w >> i), max(1, h >> i)) for i in range(levels))


def _x5(v):
    return ((v << 3) | (v >> 2)).astype(np.uint8)


def _x6(v):
    return ((v << 2) | (v >> 4)).astype(np.uint8)


def _x4(v):
    return (v * 17).astype(np.uint8)


def _x3(v):
    return ((v << 5) | (v << 2) | (v >> 1)).astype(np.uint8)


def rgb565(v):
    v = v.astype(np.uint32)
    return np.stack([_x5((v >> 11) & 0x1F), _x6((v >> 5) & 0x3F), _x5(v & 0x1F),
                     np.full(v.shape, 255, np.uint8)], -1)


def rgb5a3(v):
    v = v.astype(np.uint32)
    opaque = (v & 0x8000) != 0
    r = np.where(opaque, _x5((v >> 10) & 0x1F), _x4((v >> 8) & 0xF))
    g = np.where(opaque, _x5((v >> 5) & 0x1F), _x4((v >> 4) & 0xF))
    b = np.where(opaque, _x5(v & 0x1F), _x4(v & 0xF))
    a = np.where(opaque, 255, _x3((v >> 12) & 0x7))
    return np.stack([r, g, b, a], -1).astype(np.uint8)


def ia8(v):
    """IA8 as u16: high byte = alpha, low byte = intensity."""
    v = v.astype(np.uint32)
    i = (v & 0xFF).astype(np.uint8)
    a = (v >> 8).astype(np.uint8)
    return np.stack([i, i, i, a], -1)


TLUT_DECODE = {0: ia8, 1: rgb565, 2: rgb5a3}


def decode_palette(raw, tlut_fmt):
    """raw palette bytes -> (n, 4) RGBA array."""
    v = np.frombuffer(raw, '>u2')
    return TLUT_DECODE.get(tlut_fmt, rgb5a3)(v)


def _untile(texels, w, h, bw, bh):
    """texels: (ntiles * bh * bw, C) in GX tile order -> (h, w, C)."""
    tx, ty = -(-w // bw), -(-h // bh)
    c = texels.shape[-1]
    t = texels.reshape(ty, tx, bh, bw, c).transpose(0, 2, 1, 3, 4)
    return t.reshape(ty * bh, tx * bw, c)[:h, :w]


def _nibbles(raw):
    b = np.frombuffer(raw, np.uint8)
    return np.stack([b >> 4, b & 0xF], -1).reshape(-1)


def _cmpr_blocks(raw):
    blk = np.frombuffer(raw, np.uint8).reshape(-1, 8)
    c0 = (blk[:, 0].astype(np.uint32) << 8) | blk[:, 1]
    c1 = (blk[:, 2].astype(np.uint32) << 8) | blk[:, 3]
    p0 = rgb565(c0).astype(np.int32)
    p1 = rgb565(c1).astype(np.int32)
    four = (c0 > c1)[:, None]
    p2 = np.where(four, (p0 * 5 + p1 * 3) >> 3, (p0 + p1) >> 1)
    p3 = np.where(four, (p0 * 3 + p1 * 5) >> 3, (p0 + p1) >> 1)
    p2[:, 3] = 255
    p3[:, 3] = np.where(four[:, 0], 255, 0)
    pal = np.stack([p0, p1, p2, p3], 1).astype(np.uint8)
    bits = blk[:, 4:8]
    idx = np.stack([(bits >> s) & 3 for s in (6, 4, 2, 0)], -1).reshape(-1, 16)
    out = np.take_along_axis(pal, idx[:, :, None].astype(np.intp), 1)
    return out.reshape(-1, 4, 4, 4)


def decode_level(fmt, raw, w, h, palette=None):
    bw, bh, _ = GX_BLOCK[fmt]
    if fmt == 14:
        blocks = _cmpr_blocks(raw)
        tx, ty = -(-w // 8), -(-h // 8)
        t = blocks.reshape(ty, tx, 2, 2, 4, 4, 4).transpose(0, 2, 4, 1, 3, 5, 6)
        return t.reshape(ty * 8, tx * 8, 4)[:h, :w]
    if fmt == 6:
        t = np.frombuffer(raw, np.uint8).reshape(-1, 2, 16, 2)
        texels = np.stack([t[:, 0, :, 1], t[:, 1, :, 0], t[:, 1, :, 1], t[:, 0, :, 0]], -1)
        return _untile(texels.reshape(-1, 4), w, h, bw, bh)

    if fmt in (0, 8):
        v = _nibbles(raw)
    elif fmt in (1, 2, 9):
        v = np.frombuffer(raw, np.uint8)
    else:
        v = np.frombuffer(raw, '>u2')

    if fmt == 0:
        i = _x4(v)
        texels = np.stack([i, i, i, i], -1)
    elif fmt == 1:
        texels = np.stack([v, v, v, v], -1)
    elif fmt == 2:
        i = _x4(v & 0xF)
        a = _x4(v >> 4)
        texels = np.stack([i, i, i, a], -1)
    elif fmt == 3:
        texels = ia8(v)
    elif fmt == 4:
        texels = rgb565(v)
    elif fmt == 5:
        texels = rgb5a3(v)
    else:
        if fmt == 10:
            v = v & 0x3FFF
        if palette is None:
            i = (v * (255 // (INDEXED[fmt] - 1))).astype(np.uint8)
            texels = np.stack([i, i, i, np.full_like(i, 255)], -1)
        else:
            pal = palette
            if len(pal) < INDEXED[fmt]:
                pad = np.zeros((INDEXED[fmt] - len(pal), 4), np.uint8)
                pal = np.concatenate([pal, pad])
            texels = pal[v]
    return _untile(texels, w, h, bw, bh)


WST_HDR = 0x20


def parse_wst(data, off):
    if data[off:off + 4] != b'WST\x00' or off + WST_HDR > len(data):
        return None
    fmt, extra_mips, tlut_fmt = data[off + 0x10], data[off + 0x11], data[off + 0x12]
    w, h, pal_units, pal_size = struct.unpack_from('>HHHH', data, off + 0x14)
    data_size = struct.unpack_from('>I', data, off + 0x1c)[0]
    tex_id = struct.unpack_from('>I', data, off + 0x08)[0]
    levels = extra_mips + 1
    info = {
        'kind': 'wst', 'offset': off, 'id': tex_id, 'fmt': fmt, 'fmt_name': GX_FMT.get(fmt, f'unk{fmt}'),
        'width': w, 'height': h, 'levels': levels, 'tlut_fmt': tlut_fmt,
        'pal_entries': pal_units * 16, 'pal_size': pal_size, 'data_size': data_size,
        'pal_start': off + WST_HDR, 'pix_start': off + WST_HDR + pal_size,
        'valid': False,
    }
    if fmt in GX_BLOCK and w and h:
        expect = pal_size + mip_chain_bytes(fmt, w, h, levels)
        info['valid'] = (expect == data_size and off + WST_HDR + data_size <= len(data))
    return info


def wst_palettes(data, info):
    """Return list of (n, 4) RGBA palettes stored in the WST palette block."""
    fmt = info['fmt']
    if fmt not in INDEXED or not info['pal_size']:
        return []
    n = info['pal_entries'] or INDEXED[fmt]
    raw = data[info['pal_start']:info['pal_start'] + info['pal_size']]
    step = n * 2
    return [decode_palette(raw[i:i + step], info['tlut_fmt'])
            for i in range(0, len(raw) - step + 1, step)]


def wst_levels(data, info):
    """Yield (level, w, h, raw bytes)."""
    pos = info['pix_start']
    for lv in range(info['levels']):
        w, h = max(1, info['width'] >> lv), max(1, info['height'] >> lv)
        n = level_bytes(info['fmt'], w, h)
        yield lv, w, h, data[pos:pos + n]
        pos += n


BNTX_FMT = {
    0x02: ('R8', 1, 1, 1), 0x03: ('RGBA4', 1, 1, 2), 0x07: ('RGB565', 1, 1, 2),
    0x09: ('RG8', 1, 1, 2), 0x0b: ('RGBA8', 1, 1, 4), 0x0c: ('BGRA8', 1, 1, 4),
    0x1a: ('BC1', 4, 4, 8), 0x1b: ('BC2', 4, 4, 16), 0x1c: ('BC3', 4, 4, 16),
    0x1d: ('BC4', 4, 4, 8), 0x1e: ('BC5', 4, 4, 16), 0x1f: ('BC6H', 4, 4, 16),
    0x20: ('BC7', 4, 4, 16),
}
_BCN_PIL = {'BC1': (1, 'BC1', 'RGBA'), 'BC2': (2, 'BC2', 'RGBA'), 'BC3': (3, 'BC3', 'RGBA'),
            'BC4': (4, 'BC4', 'L'), 'BC5': (5, 'BC5', 'RGB'), 'BC6H': (6, 'BC6H', 'RGB'),
            'BC7': (7, 'BC7', 'RGBA')}


def _u16(d, o):
    return struct.unpack_from('<H', d, o)[0]


def _u32(d, o):
    return struct.unpack_from('<I', d, o)[0]


def _u64(d, o):
    return struct.unpack_from('<Q', d, o)[0]


def parse_bntx(data, base):
    if data[base:base + 4] != b'BNTX' or data[base + 0x20:base + 0x24] != b'NX  ':
        return
    count, arr = _u32(data, base + 0x24), _u64(data, base + 0x28)
    for i in range(count):
        bo = base + _u64(data, base + arr + i * 8)
        if data[bo:bo + 4] != b'BRTI':
            continue
        fmt = _u32(data, bo + 0x1c)
        w, h, depth, layers, bh_log2 = struct.unpack_from('<5I', data, bo + 0x24)
        mips = _u16(data, bo + 0x16)
        name_ptr = _u64(data, bo + 0x60)
        nlen = _u16(data, base + name_ptr)
        name = data[base + name_ptr + 2:base + name_ptr + 2 + nlen].decode('utf-8', 'replace')
        mip_arr = base + _u64(data, bo + 0x70)
        fmt_name = BNTX_FMT.get(fmt >> 8, (f'unk{fmt:04x}',))[0]
        yield {
            'kind': 'bntx', 'offset': bo, 'name': name, 'fmt': fmt, 'fmt_name': fmt_name,
            'width': w, 'height': h, 'levels': mips, 'layers': layers,
            'tile_mode': _u16(data, bo + 0x12), 'block_height_log2': bh_log2 & 7,
            'image_size': _u32(data, bo + 0x50),
            'channels': list(data[bo + 0x58:bo + 0x5c]),
            'mip_offsets': [base + _u64(data, mip_arr + m * 8) for m in range(mips)],
            'valid': (fmt >> 8) in BNTX_FMT and (_u16(data, bo + 0x12) == 0) and w > 0 and h > 0,
        }


def _deswizzle_block_linear(raw, wb, hb, bpb, bh_log2):
    gob_h = 8 << bh_log2
    row_bytes = wb * bpb
    gobs_x = -(-row_bytes // 64)
    x = np.arange(row_bytes, dtype=np.int64)[None, :]
    y = np.arange(hb, dtype=np.int64)[:, None]
    addr = ((y // gob_h) * 512 * (1 << bh_log2) * gobs_x
            + (x // 64) * 512 * (1 << bh_log2)
            + ((y % gob_h) // 8) * 512
            + ((x % 64) // 32) * 256 + ((y % 8) // 2) * 64
            + ((x % 32) // 16) * 32 + (y % 2) * 16 + (x % 16))
    src = np.frombuffer(raw, np.uint8)
    if addr.max() >= len(src):
        src = np.concatenate([src, np.zeros(int(addr.max()) + 1 - len(src), np.uint8)])
    return src[addr].tobytes()


def bntx_levels(data, info, layer=0):
    _, bw, bh, bpb = BNTX_FMT[info['fmt'] >> 8]
    layer_size = info['image_size'] // max(1, info['layers'])
    for lv, off in enumerate(info['mip_offsets']):
        w, h = max(1, info['width'] >> lv), max(1, info['height'] >> lv)
        wb, hb = -(-w // bw), -(-h // bh)
        lg = info['block_height_log2']
        while lg > 0 and hb <= (1 << (lg - 1)) * 8:
            lg -= 1
        start = off + layer * layer_size
        gob_h = 8 << lg
        size = (-(-wb * bpb // 64) * 64) * (-(-hb // gob_h) * gob_h)
        yield lv, w, h, _deswizzle_block_linear(data[start:start + size], wb, hb, bpb, lg)


def decode_bntx_level(info, raw, w, h):
    kind, bw, bh, bpb = BNTX_FMT[info['fmt'] >> 8]
    if kind in _BCN_PIL:
        n, pf, mode = _BCN_PIL[kind]
        pw, ph = -(-w // 4) * 4, -(-h // 4) * 4
        im = Image.frombytes(mode, (pw, ph), raw, 'bcn', (n, pf if (info['fmt'] & 0xff) != 2 else pf + 'S'))
        a = np.asarray(im)
        if mode == 'L':
            a = np.stack([a, np.zeros_like(a), np.zeros_like(a), np.full_like(a, 255)], -1)
        elif mode == 'RGB':
            a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
        src = a[:h, :w]
    else:
        v = np.frombuffer(raw, np.uint8)[:w * h * bpb].reshape(h, w, bpb)
        z, f = np.zeros((h, w), np.uint8), np.full((h, w), 255, np.uint8)
        if kind == 'R8':
            src = np.stack([v[..., 0], z, z, f], -1)
        elif kind == 'RG8':
            src = np.stack([v[..., 0], v[..., 1], z, f], -1)
        elif kind == 'RGBA8':
            src = v
        elif kind == 'BGRA8':
            src = v[..., [2, 1, 0, 3]]
        elif kind == 'RGB565':
            src = rgb565(v[..., 0].astype(np.uint16) | (v[..., 1].astype(np.uint16) << 8))
        else:
            u = v[..., 0].astype(np.uint16) | (v[..., 1].astype(np.uint16) << 8)
            src = np.stack([_x4(u & 0xF), _x4((u >> 4) & 0xF), _x4((u >> 8) & 0xF), _x4(u >> 12)], -1)
    sel = info['channels']
    if sel != [2, 3, 4, 5] and all(c <= 5 for c in sel):
        ext = np.concatenate([np.zeros(src.shape[:2] + (1,), np.uint8),
                              np.full(src.shape[:2] + (1,), 255, np.uint8), src], -1)
        src = ext[..., sel]
    return np.ascontiguousarray(src, np.uint8)

def _cstr(b):
    return b.split(b'\0', 1)[0].decode('ascii', 'replace')


def walk_arcl(data, base, end, path):
    e = '>' if struct.unpack_from('>I', data, base + 8)[0] < 0x10000 else '<'
    count, hsize = struct.unpack_from(e + 'II', data, base + 4)
    for i in range(count):
        eo = base + struct.unpack_from(e + 'I', data, base + hsize + i * 4)[0]
        size, doff = struct.unpack_from(e + 'II', data, eo)
        if size == 0:
            continue
        yield from walk(data, base + doff, base + doff + size, path + [f'{i:03d}'])


def walk_wpt(data, base, end, path):
    count = struct.unpack_from('>I', data, base + 0x10)[0]
    offs = struct.unpack_from(f'>{count}I', data, base + 0x14)
    for i, o in enumerate(offs):
        if o:
            yield from walk(data, base + 0x10 + o, end, path + [f'{i:03d}'])


def walk_sarc(data, base, end, path):
    data_off = _u32(data, base + 0x0c)
    nodes = _u16(data, base + 0x1a)
    names = base + 0x20 + nodes * 16 + 8
    for i in range(nodes):
        n = base + 0x20 + i * 16
        attr, s, e = _u32(data, n + 4), _u32(data, n + 8), _u32(data, n + 12)
        name = _cstr(data[names + (attr & 0xFFFFFF) * 4:names + (attr & 0xFFFFFF) * 4 + 256]) \
            if attr >> 24 else f'{i:03d}'
        label = os.path.splitext(name.replace('/', '_'))[0]
        yield from walk(data, base + data_off + s, base + data_off + e, path + [label])


def walk_bntx(data, base, end, path):
    for info in parse_bntx(data, base):
        yield path + [info['name']], info


def walk_cmp(data, base, end, path):
    count = struct.unpack_from('>I', data, base + 4)[0]
    ents = [struct.unpack_from('>II', data, base + 8 + i * 8) for i in range(count)]
    for i, (eid, o) in enumerate(ents):
        nxt = ents[i + 1][1] if i + 1 < len(ents) else end - base
        yield from walk(data, base + o, base + nxt, path + [f'{i:03d}_{eid:08x}'])


def walk_menu(data, base, end, path):
    pos = base + 0x18
    while pos + 0x28 <= end and data[pos:pos + 3] != b'EOF':
        typ = _cstr(data[pos:pos + 4])
        name = _cstr(data[pos + 6:pos + 0x24])
        size = struct.unpack_from('>I', data, pos + 0x24)[0]
        if size:
            label = f'{typ}_{os.path.splitext(name)[0]}'
            yield from walk(data, pos + 0x28, pos + 0x28 + size, path + [label])
        pos += 0x28 + size


def walk_scan(data, base, end, path):
    n = 0
    pos = base
    while True:
        a = data.find(b'WST\x00', pos, end)
        b = data.find(b'BNTX\x00\x00\x00\x00', pos, end)
        hits = [h for h in (a, b) if h != -1]
        if not hits:
            return
        pos = min(hits)
        if pos == b:
            texs = list(parse_bntx(data, pos))
            for info in texs:
                yield path + [f'{n:03d}', info['name']], info
            n += 1
            pos += max(4, _u32(data, pos + 0x1c)) if texs else 4
        else:
            info = parse_wst(data, pos)
            if info and info['valid']:
                yield path + [f'{n:03d}'], info
                n += 1
                pos += WST_HDR + info['data_size']
            else:
                pos += 4


CONTAINERS = {b'ArcL': walk_arcl, b'WPT\x00': walk_wpt, b'CMP\x00': walk_cmp, b'Menu': walk_menu,
              b'SARC': walk_sarc, b'BNTX': walk_bntx}
CONTAINER_NAMES = {b'ArcL': 'ArcL', b'WPT\x00': 'WPT', b'CMP\x00': 'CMP', b'Menu': 'Menu',
                   b'SARC': 'SARC', b'BNTX': 'BNTX', b'WST\x00': 'WST'}


def texture_table(data, base=0):
    if base + 8 > len(data):
        return None
    count = struct.unpack_from('>I', data, base)[0]
    if not 0 < count < 0x4000 or base + 4 + count * 4 > len(data):
        return None
    offs = [v & 0xFFFFFF for v in struct.unpack_from(f'>{count}I', data, base + 4)]
    if any(offs) and all(data[base + o:base + o + 4] == b'WST\x00' for o in offs if o):
        return offs
    return None


def walk_texture_table(data, base, end, path):
    for i, o in enumerate(texture_table(data, base)):
        if o:
            info = parse_wst(data, base + o)
            if info:
                yield path + [f'{i:03d}'], info


def walk(data, base=0, end=None, path=None, scan_unknown=False):
    end = len(data) if end is None else end
    path = path or []
    magic = data[base:base + 4]
    if magic == b'WST\x00':
        info = parse_wst(data, base)
        if info:
            yield path, info
    elif magic in CONTAINERS:
        yield from CONTAINERS[magic](data, base, end, path)
    elif base == 0 and texture_table(data):
        yield from walk_texture_table(data, base, end, path)
    elif scan_unknown:
        yield from walk_scan(data, base, end, path)


def container_type(data):
    if data[:4] not in CONTAINER_NAMES and texture_table(data):
        return 'TexTable'
    return CONTAINER_NAMES.get(data[:4])


def _unique_palettes(pals):
    """Drop duplicate palettes and filler palettes where every entry is the same."""
    out, seen = [], set()
    for i, p in enumerate(pals):
        key = p.tobytes()
        if key in seen or (i > 0 and (p == p[0]).all()):
            continue
        seen.add(key)
        out.append((i, p))
    return out


def _save(rgba, out_base, suffix):
    path = out_base.with_name(out_base.name + suffix + '.png')
    Image.fromarray(np.ascontiguousarray(rgba), 'RGBA').save(path)
    return path


def extract_bntx(data, info, out_base, all_mips=False):
    saved = []
    for layer in range(info['layers']):
        for lv, w, h, raw in bntx_levels(data, info, layer):
            if lv > 0 and not all_mips:
                break
            suffix = (f'_layer{layer}' if info['layers'] > 1 else '') + (f'_mip{lv}' if lv else '')
            saved.append(_save(decode_bntx_level(info, raw, w, h), out_base, suffix))
    return saved


def extract_texture(data, info, out_base, palettes='unique', all_mips=False):
    if info['kind'] == 'bntx':
        return extract_bntx(data, info, out_base, all_mips)
    fmt = info['fmt']
    saved = []
    pals = wst_palettes(data, info)
    if fmt in INDEXED:
        if not pals:
            choices = [(None, None)]
        elif palettes == 'first':
            choices = [(0, pals[0])]
        elif palettes == 'all':
            choices = list(enumerate(pals))
        else:
            choices = _unique_palettes(pals)
    else:
        choices = [(None, None)]

    for lv, w, h, raw in wst_levels(data, info):
        if lv > 0 and not all_mips:
            break
        for pi, pal in choices:
            suffix = (f'_mip{lv}' if lv else '') + (f'_p{pi}' if pi is not None and len(choices) > 1 else '')
            saved.append(_save(decode_level(fmt, raw, w, h, pal), out_base, suffix))
    return saved


def describe(path_parts, info):
    where = '/'.join(path_parts) or '-'
    if info['kind'] == 'bntx':
        extra = f" layers={info['layers']}" if info['layers'] > 1 else ''
        flag = '' if info['valid'] else '  UNSUPPORTED'
        return (f"  {where:40s} {info['width']:4d}x{info['height']:<4d} {info['fmt_name']:6s} "
                f"levels={info['levels']:<2d} fmt={info['fmt']:04x}{extra}{flag}")
    pal = ''
    if info['fmt'] in INDEXED:
        npal = info['pal_size'] // max(2, info['pal_entries'] * 2)
        pal = f"tlut={TLUT_FMT.get(info['tlut_fmt'], '?')} x{npal}"
    elif info['pal_size']:
        pal = f"extra={info['pal_size']:#x}"
    flag = '' if info['valid'] else '  SIZE MISMATCH'
    return (f"  {where:24s} {info['width']:4d}x{info['height']:<4d} "
            f"{info['fmt_name']:6s} levels={info['levels']:<2d} id={info['id']:08x} {pal}{flag}")


def process_file(src, out_dir, args, rel_name=None):
    data = src.read_bytes()
    ctype = container_type(data)
    if ctype is None and not args.scan:
        return None
    label = rel_name or src.name
    try:
        entries = list(walk(data, scan_unknown=args.scan))
    except (struct.error, IndexError, ValueError) as e:
        print(f"{label}: {ctype or 'scan'}, ERROR while reading container: {e}")
        return {'textures': 0, 'images': 0, 'errors': 1}

    if args.list:
        print(f"{label}  -  {ctype or 'scan'}, {len(entries)} texture(s)")
        for parts, info in entries:
            print(describe(parts, info))
        return {'textures': len(entries), 'images': 0, 'errors': 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, images, errors = [], 0, 0
    for parts, info in entries:
        stem = '_'.join(parts) if parts else '000'
        stem += f"_{info['width']}x{info['height']}_{info['fmt_name']}"
        stem = ''.join(c if c.isalnum() or c in '._-' else '_' for c in stem)
        rec = {k: v for k, v in info.items() if k not in ('pal_start', 'pix_start', 'mip_offsets')}
        rec['path'] = '/'.join(parts)
        try:
            if not info['valid']:
                raise ValueError('unsupported format or header sizes do not match, skipped')
            saved = extract_texture(data, info, out_dir / stem, args.palettes, args.mips)
            rec['files'] = [p.name for p in saved]
            images += len(saved)
            if args.verbose:
                print(describe(parts, info))
        except Exception as e:
            rec['error'] = str(e)
            errors += 1
            print(f"  {label}: {'/'.join(parts)}  ERROR: {e}")
        manifest.append(rec)
    if manifest:
        (out_dir / '_manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f"{label}: {ctype or 'scan'}, {len(entries)} texture(s), {images} image(s)"
          + (f", {errors} error(s)" if errors else ''))
    return {'textures': len(entries), 'images': images, 'errors': errors}


def main():
    ap = argparse.ArgumentParser(
        description='Extract textures from Go Vacation (Wii / Switch) files: '
                    'ArcL / WPT / CMP / Menu / WST (Wii) and SARC / BNTX (Switch).')
    ap.add_argument('input', help='TEX file, or a folder (searched recursively)')
    ap.add_argument('output', nargs='?', default=None,
                    help='Output directory (default: <input>_out)')
    ap.add_argument('-l', '--list', action='store_true', help='List textures without extracting')
    ap.add_argument('-v', '--verbose', action='store_true', help='Print every texture')
    ap.add_argument('--palettes', choices=['unique', 'first', 'all'], default='unique',
                    help='Which palettes of C4/C8 textures to export (default: unique)')
    ap.add_argument('--mips', action='store_true', help='Also export mipmap levels')
    ap.add_argument('--all-files', action='store_true',
                    help='Folder mode: process every file with a known magic, not only "TEX"')
    ap.add_argument('--scan', action='store_true',
                    help='Search unknown containers for embedded WST / BNTX textures')
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output) if args.output else src.parent / (src.name + '_out')

    if src.is_file():
        if process_file(src, out, args) is None:
            print(f"Error: {src} - unknown container (magic={src.read_bytes()[:4]!r}); try --scan")
            sys.exit(1)
        return

    totals = {'files': 0, 'textures': 0, 'images': 0, 'errors': 0}
    empty_tex = 0
    for f in sorted(p for p in src.rglob('*') if p.is_file()):
        if not args.all_files and f.name.upper() != 'TEX':
            continue
        if f.stat().st_size == 0:
            empty_tex += f.name.upper() == 'TEX'
            continue
        rel = f.relative_to(src)
        r = process_file(f, out / rel.parent / rel.name, args, rel_name=str(rel))
        if r:
            totals['files'] += 1
            for k in ('textures', 'images', 'errors'):
                totals[k] += r[k]
    print(f"\nDone - {totals['files']} file(s), {totals['textures']} texture(s), "
          f"{totals['images']} image(s), {totals['errors']} error(s)")
    if empty_tex and not args.all_files:
        print(f"Note: {empty_tex} TEX file(s) are empty. This looks like the Switch dump, where "
              f"textures live in MAPMODEL / ARC / CUBETEX / EFF ... files - use --all-files --scan.")


if __name__ == '__main__':
    main()
