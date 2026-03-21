import sys
import os
import struct
import math
import argparse
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required.  pip install Pillow")
    sys.exit(1)


# GX colour helpers 

def _rgb565(v):
    r = ((v >> 11) & 0x1F) * 255 // 31
    g = ((v >>  5) & 0x3F) * 255 // 63
    b = ( v        & 0x1F) * 255 // 31
    return (r, g, b, 255)


def _rgb5a3(v):
    if v & 0x8000:                              # RGB555
        r = ((v >> 10) & 0x1F) * 255 // 31
        g = ((v >>  5) & 0x1F) * 255 // 31
        b = ( v        & 0x1F) * 255 // 31
        return (r, g, b, 255)
    else:                                       # RGB4A3 
        a = ((v >> 12) & 0x7) * 255 // 7
        r = ((v >>  8) & 0xF) * 255 // 15
        g = ((v >>  4) & 0xF) * 255 // 15
        b = ( v        & 0xF) * 255 // 15
        return (r, g, b, a)


# Mip-chain size 

_FMT_PARAMS = {
    5:  (4, 4, 2),      # RGB5A3
    6:  (4, 4, 4),      # RGBA8
    8:  (8, 8, 0.5),    # C4
    9:  (8, 4, 1.0),    # C8
    14: (8, 8, 0.5),    # CMPR
}

def mip_chain_bytes(w, h, mips, fmt):
    tw, th, bpp = _FMT_PARAMS.get(fmt, (4, 4, 1))
    total = 0
    for m in range(mips):
        mw = max(tw, math.ceil(max(1, w >> m) / tw) * tw)
        mh = max(th, math.ceil(max(1, h >> m) / th) * th)
        total += int(mw * mh * bpp)
    return total


# Format decoders 

def decode_cmpr(pixels, w, h):
    """
    CMPR = GX variant of DXT1.
    Layout: 8x8 super-tiles each containing four 4x4 DXT1 sub-blocks.
    Colour order in each sub-block is SWAPPED vs PC DXT1 (c0/c1 reversed
    for the transparent/black entry selection).
    """
    out = bytearray(w * h * 4)
    tiles_x = max(1, (w + 7) // 8)
    tiles_y = max(1, (h + 7) // 8)
    pos = 0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            for sub_y in range(2):
                for sub_x in range(2):
                    blk = pixels[pos:pos + 8]; pos += 8
                    c0 = struct.unpack_from('>H', blk, 0)[0]
                    c1 = struct.unpack_from('>H', blk, 2)[0]
                    lut = struct.unpack_from('>I', blk, 4)[0]

                    def e565(c):
                        r = ((c >> 11) & 0x1F) * 255 // 31
                        g = ((c >>  5) & 0x3F) * 255 // 63
                        b = ( c        & 0x1F) * 255 // 31
                        return (r, g, b, 255)

                    col = [e565(c0), e565(c1)]
                    if c0 > c1:
                        col.append(tuple((2*a + b) // 3 for a, b in zip(col[0], col[1])))
                        col.append(tuple((a + 2*b) // 3 for a, b in zip(col[0], col[1])))
                    else:
                        col.append(tuple((a + b) // 2 for a, b in zip(col[0], col[1])))
                        col.append((0, 0, 0, 0))   # transparent

                    for py in range(4):
                        for px in range(4):
                            bit_pos = (py * 4 + px) * 2
                            idx = (lut >> (30 - bit_pos)) & 3
                            ox = tx * 8 + sub_x * 4 + px
                            oy = ty * 8 + sub_y * 4 + py
                            if ox < w and oy < h:
                                o = (oy * w + ox) * 4
                                out[o:o + 4] = col[idx]
    return bytes(out)


def decode_rgb5a3_direct(pixels, w, h):
    """RGB5A3: 4x4 tiles, 2 bytes per pixel."""
    out = bytearray(w * h * 4)
    tiles_x = max(1, (w + 3) // 4)
    tiles_y = max(1, (h + 3) // 4)
    pos = 0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = pixels[pos:pos + 32]; pos += 32
            for py in range(4):
                for px in range(4):
                    ni = py * 4 + px
                    v = struct.unpack_from('>H', tile, ni * 2)[0] if ni * 2 + 1 < len(tile) else 0
                    col = _rgb5a3(v)
                    ox = tx * 4 + px; oy = ty * 4 + py
                    if ox < w and oy < h:
                        o = (oy * w + ox) * 4
                        out[o:o + 4] = col
    return bytes(out)


def decode_rgba8(pixels, w, h):
    """RGBA8: 4x4 tiles, 32 bytes per tile (split AR / GB planes)."""
    out = bytearray(w * h * 4)
    tiles_x = max(1, (w + 3) // 4)
    tiles_y = max(1, (h + 3) // 4)
    pos = 0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = pixels[pos:pos + 64]; pos += 64
            for py in range(4):
                for px in range(4):
                    ni = py * 4 + px
                    a = tile[ni * 2]
                    r = tile[ni * 2 + 1]
                    g = tile[32 + ni * 2]
                    b = tile[32 + ni * 2 + 1]
                    ox = tx * 4 + px; oy = ty * 4 + py
                    if ox < w and oy < h:
                        o = (oy * w + ox) * 4
                        out[o:o + 4] = (r, g, b, a)
    return bytes(out)


def decode_c4(pixels, palette_u16, w, h, tlut_fmt):
    """C4: 8x8 tiles, 4-bit indexed."""
    dec = _rgb565 if tlut_fmt == 1 else _rgb5a3
    out = bytearray(w * h * 4)
    tiles_x = max(1, (w + 7) // 8)
    tiles_y = max(1, (h + 7) // 8)
    pos = 0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = pixels[pos:pos + 32]; pos += 32
            for py in range(8):
                for px in range(8):
                    ni = py * 8 + px
                    byte = tile[ni // 2] if ni // 2 < len(tile) else 0
                    idx = (byte >> 4) if (ni % 2 == 0) else (byte & 0xF)
                    col = dec(palette_u16[idx]) if idx < len(palette_u16) else (255, 0, 255, 255)
                    ox = tx * 8 + px; oy = ty * 8 + py
                    if ox < w and oy < h:
                        o = (oy * w + ox) * 4
                        out[o:o + 4] = col
    return bytes(out)


def decode_c8(pixels, palette_u16, w, h, tlut_fmt):
    """C8: 8x4 tiles, 8-bit indexed."""
    dec = _rgb565 if tlut_fmt == 1 else _rgb5a3
    out = bytearray(w * h * 4)
    tiles_x = max(1, (w + 7) // 8)
    tiles_y = max(1, (h + 3) // 4)
    pos = 0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = pixels[pos:pos + 32]; pos += 32
            for py in range(4):
                for px in range(8):
                    ni = py * 8 + px
                    byte = tile[ni] if ni < len(tile) else 0
                    col = dec(palette_u16[byte]) if byte < len(palette_u16) else (255, 0, 255, 255)
                    ox = tx * 8 + px; oy = ty * 4 + py
                    if ox < w and oy < h:
                        o = (oy * w + ox) * 4
                        out[o:o + 4] = col
    return bytes(out)


# WST reader (shared by ArcL and WPT)

GX_FMT = {
    0: 'I4', 1: 'I8', 2: 'IA4', 3: 'IA8',
    4: 'RGB565', 5: 'RGB5A3', 6: 'RGBA8',
    8: 'C4', 9: 'C8', 10: 'C14X2', 14: 'CMPR',
}
TLUT_FMT = {0: 'IA8', 1: 'RGB565', 2: 'RGB5A3'}

PALETTE_FREE   = {5, 6, 14}
PALETTE_NEEDED = {8, 9}


def read_wst(data, wst_off):
    """
    Parse a WST header at wst_off and return an info dict, or None if no WST magic.
    tex_start = wst_off + 0x20 (pixel data begins right after the 32-byte WST header).
    """
    wst = data[wst_off:]
    if wst[:4] != b'WST\x00':
        return None

    fmt      = wst[0x10]
    mips     = wst[0x11]
    npal     = wst[0x12]
    width    = struct.unpack_from('>H', wst, 0x14)[0]
    height   = struct.unpack_from('>H', wst, 0x16)[0]
    tlut_fmt = struct.unpack_from('>H', wst, 0x18)[0]
    tlut_cnt = struct.unpack_from('>H', wst, 0x1a)[0]
    data_sz  = struct.unpack_from('>I', wst, 0x1c)[0]

    tex_start = wst_off + 0x20
    mc        = mip_chain_bytes(width, height, mips, fmt)
    pal_data  = data[tex_start + mc : tex_start + data_sz]

    return {
        'fmt':       fmt,
        'fmt_name':  GX_FMT.get(fmt, f'unk{fmt:#x}'),
        'mips':      mips,
        'npal':      npal,
        'width':     width,
        'height':    height,
        'tlut_fmt':  tlut_fmt,
        'tlut_cnt':  tlut_cnt,
        'data_sz':   data_sz,
        'tex_start': tex_start,
        'mip_bytes': mc,
        'pal_data':  pal_data,
    }


# ArcL parser

def parse_arcl(data):
    """
    Yield (index, info) for each WST entry in an ArcL archive.
    ArcL entry layout: 0x20 bytes pre-WST, then WST header at entry_offset+0x20.
    """
    count   = struct.unpack_from('>I', data, 4)[0]
    offsets = [struct.unpack_from('>I', data, 0x20 + i * 4)[0] for i in range(count)]

    for idx, off in enumerate(offsets):
        info = read_wst(data, off + 0x20)
        yield idx, info


# WPT parser

def parse_wpt(data):
    """
    Yield (index, info) for each WST entry in a WPT archive.

    WPT header layout:
      0x00  magic "WPT\0"
      0x04  u32 version (1)
      0x10  u32 offset_table_start  (always 0x20)
      0x14  u32 offset_table_end    (also = section1_ptr, WST at +0x10)
      0x18  u32 section2_ptr        (WST at +0x10)
      0x1c  u32 section3_ptr        (WST at +0x10)

    All entries (sections and offset table) have 0x10 bytes pre-WST,
    so WST header is always at entry_pointer + 0x10.
    """
    tbl_start = struct.unpack_from('>I', data, 0x10)[0]
    tbl_end   = struct.unpack_from('>I', data, 0x14)[0]
    sec2_ptr  = struct.unpack_from('>I', data, 0x18)[0]
    sec3_ptr  = struct.unpack_from('>I', data, 0x1c)[0]

    
    wst_ptrs = [tbl_end, sec2_ptr, sec3_ptr]
    count = (tbl_end - tbl_start) // 4
    for i in range(count):
        v = struct.unpack_from('>I', data, tbl_start + i * 4)[0]
        if v != 0:
            wst_ptrs.append(v)

    for idx, ptr in enumerate(wst_ptrs):
        info = read_wst(data, ptr + 0x10)
        yield idx, info




def parse_file(data):
    """Auto-detect ArcL or WPT and return (container_type, generator)."""
    magic = data[:4]
    if magic == b'ArcL':
        return 'ArcL', parse_arcl(data)
    elif magic == b'WPT\x00':
        return 'WPT', parse_wpt(data)
    else:
        raise ValueError(f"Unknown format (magic={magic!r})")


# Entry extractor

def extract_entry(data, idx, info, out_dir, skip_indexed=False, verbose=False):
    """Decode one WST entry and save PNG(s). Returns list of saved paths."""
    fmt      = info['fmt']
    w        = info['width']
    h        = info['height']
    mips     = info['mips']
    tlut_fmt = info['tlut_fmt']
    tlut_cnt = info['tlut_cnt']
    mc       = info['mip_bytes']
    ts       = info['tex_start']
    pal_data = info['pal_data']
    fname    = info['fmt_name']

    if verbose:
        pal_note = f"tlut={TLUT_FMT.get(tlut_fmt,'?')}({tlut_cnt})" if fmt in PALETTE_NEEDED else ""
        print(f"  [{idx:3d}] {w:4d}x{h:<4d}  {fname:7s}  mips={mips}  {pal_note}")

    pixels = data[ts : ts + mc]
    saved  = []

    if mc == 0 or w == 0 or h == 0:
        return saved  

    # palette-free formats
    if fmt == 14:   # CMPR
        rgba = decode_cmpr(pixels, w, h)
        path = out_dir / f"{idx:03d}_{w}x{h}_CMPR.png"
        Image.frombytes('RGBA', (w, h), rgba).save(path)
        saved.append(path)

    elif fmt == 5:  # RGB5A3
        rgba = decode_rgb5a3_direct(pixels, w, h)
        path = out_dir / f"{idx:03d}_{w}x{h}_RGB5A3.png"
        Image.frombytes('RGBA', (w, h), rgba).save(path)
        saved.append(path)

    elif fmt == 6:  # RGBA8
        rgba = decode_rgba8(pixels, w, h)
        path = out_dir / f"{idx:03d}_{w}x{h}_RGBA8.png"
        Image.frombytes('RGBA', (w, h), rgba).save(path)
        saved.append(path)

    # palette-indexed formats
    elif fmt == 8 and not skip_indexed:   # C4
        n_raw     = len(pal_data) // 2
        n_pal     = max(1, min(tlut_cnt // 16 if tlut_cnt else n_raw // 16, n_raw // 16))
        all_pal   = [struct.unpack_from('>H', pal_data, j * 2)[0] for j in range(n_raw)]
        for p in range(n_pal):
            pal16 = (all_pal[p * 16 : p * 16 + 16] + [0] * 16)[:16]
            rgba = decode_c4(pixels, pal16, w, h, tlut_fmt)
            suf  = f"_p{p}" if n_pal > 1 else ""
            path = out_dir / f"{idx:03d}_{w}x{h}_C4{suf}.png"
            Image.frombytes('RGBA', (w, h), rgba).save(path)
            saved.append(path)

    elif fmt == 9 and not skip_indexed:   # C8
        PAL  = 512  # 256 entries x 2 bytes
        n_av = max(1, len(pal_data) // PAL)
        n_dc = max(1, tlut_cnt // 256) if tlut_cnt >= 256 else 1
        n_pal = min(n_av, n_dc)
        for p in range(n_pal):
            raw  = pal_data[p * PAL : p * PAL + PAL]
            pal  = ([struct.unpack_from('>H', raw, j * 2)[0] for j in range(len(raw) // 2)]
                    + [0] * 256)[:256]
            rgba = decode_c8(pixels, pal, w, h, tlut_fmt)
            suf  = f"_p{p}" if n_pal > 1 else ""
            path = out_dir / f"{idx:03d}_{w}x{h}_C8{suf}.png"
            Image.frombytes('RGBA', (w, h), rgba).save(path)
            saved.append(path)

    return saved


# CLI 

def main():
    ap = argparse.ArgumentParser(
        description='Extract textures from a Go Vacation ArcL or WPT/WST file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=None
    )
    ap.add_argument('input',  help='ArcL or WPT file to extract (e.g. TEX)')
    ap.add_argument('output', nargs='?', default=None,
                    help='Output directory (default: <input>_out/)')
    ap.add_argument('-l', '--list',      action='store_true',
                    help='List contents without extracting')
    ap.add_argument('--cmpr-only',       action='store_true',
                    help='Skip palette-indexed (C4/C8) entries')
    ap.add_argument('-v', '--verbose',   action='store_true',
                    help='Print header info for every entry')
    args = ap.parse_args()

    src  = Path(args.input)
    data = src.read_bytes()

    try:
        container, entries = parse_file(data)
    except ValueError as e:
        print(f"Error: {src} - {e}")
        sys.exit(1)

    # consume generator into list so we can count
    entry_list = [(idx, info) for idx, info in entries if info is not None]
    count = len(entry_list)

    if args.list:
        print(f"{src.name}  -  {container} archive, {count} entries\n")
        fmt_counts = {}
        for idx, info in entry_list:
            tag = info['fmt_name']
            fmt_counts[tag] = fmt_counts.get(tag, 0) + 1
            pf  = 'Working' if info['fmt'] in PALETTE_FREE else 'Warning: Needs palette'
            print(f"  [{idx:3d}]  {info['width']:4d}x{info['height']:<4d}  "
                  f"{tag:7s}  mips={info['mips']}  {pf}")
        print(f"\nSummary: {', '.join(f'{v}x{k}' for k,v in sorted(fmt_counts.items()))}")
        return

    out_dir = Path(args.output) if args.output else src.parent / (src.name + '_out')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{container}: {count} entries  ->  {out_dir}/")

    totals = {'ok': 0, 'skip': 0, 'err': 0}
    fmt_counts = {}

    for idx, info in entry_list:
        fmt = info['fmt']
        fmt_counts[info['fmt_name']] = fmt_counts.get(info['fmt_name'], 0) + 1

        if fmt not in PALETTE_FREE and fmt not in PALETTE_NEEDED:
            if args.verbose:
                print(f"  [{idx:3d}]  {info['fmt_name']} - unsupported, skipping")
            totals['skip'] += 1
            continue

        if fmt in PALETTE_NEEDED and args.cmpr_only:
            totals['skip'] += 1
            continue

        try:
            saved = extract_entry(data, idx, info, out_dir,
                                  skip_indexed=args.cmpr_only,
                                  verbose=args.verbose)
            totals['ok'] += len(saved)
            if not args.verbose and saved:
                print(f"  [{idx:3d}]  {', '.join(p.name for p in saved)}")
        except Exception as e:
            print(f"  [{idx:3d}]  ERROR: {e}")
            totals['err'] += 1

    print(f"\nDone  -  {totals['ok']} image(s) saved"
          + (f", {totals['skip']} skipped" if totals['skip'] else "")
          + (f", {totals['err']} error(s)" if totals['err'] else ""))
    if fmt_counts:
        print("Format breakdown: " + ', '.join(f"{v}x{k}" for k,v in sorted(fmt_counts.items())))


if __name__ == '__main__':
    main()
