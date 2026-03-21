

Tool for extracting textures from Go Vacation (Wii) TEX files.

A tool for working with texture archives from Go Vacation (Wii). Supports extracting ArcL archives containing WST-wrapped GX textures, including CMPR, RGB5A3, and palette-indexed formats. Only for educational purposes


# Warning

# This is an experimental tool which is not yet completed. It's currently only for testing purposes. A proper writeup will probably follow when im done understanding all the file formats.


---

## Format support

| Format | Decodes? | Colours correct?           |
| ------ | -------- | -------------------------- |
| CMPR   | Yes      | Yes                        |
| RGB5A3 | Yes      | Yes                        |
| RGBA8  | Yes      | Yes                        |
| C4     | Yes      | No but im working on a fix |
| C8     | Yes      | No but im working on a fix |

---

## File format reference

```
ArcL archive
├── 0x00  magic "ArcL"
├── 0x04  u32 BE  entry count
├── 0x20  u32[count]  entry offsets (BE)
└── entries (packed back-to-back, no gaps):
    ├── +0x00  u32 BE  inner_size  (= data_sz + 0x20)
    ├── +0x04  u32 BE  base_offset (= absolute offset of WST header)
    ├── +0x08  8 × 0xff  padding
    ├── +0x10  16 × 0x00  padding
    └── +0x20  WST header (32 bytes)
        ├── +0x00  magic "WST\0"
        ├── +0x10  u8   gx_format  (5=RGB5A3, 6=RGBA8, 8=C4, 9=C8, 14=CMPR)
        ├── +0x11  u8   mip_count
        ├── +0x12  u8   npal
        ├── +0x14  u16 BE  width
        ├── +0x16  u16 BE  height
        ├── +0x18  u16 BE  tlut_format  (0=IA8, 1=RGB565, 2=RGB5A3)
        ├── +0x1a  u16 BE  tlut_count
        └── +0x1c  u32 BE  data_size  (pixel bytes + palette bytes)
            pixel data starts at WST+0x20, palette data follows immediately after
```

### CMPR tile layout
- 8×8 super-tiles each containing four 4×4 DXT1 sub-blocks
- Each sub-block: 2× RGB565 colours + 32-bit 2bpp index table
- Colour interpolation: if c0 > c1 → 4-colour mode, else 3-colour + transparent
- GX difference from PC DXT1: same encoding, just arranged in 8×8 super-tiles

---

## Credits

- [RRUnpacker](https://github.com/Nenkai/RRUnpacker) by Nenkai - without this tool none of this would be accessible
- GX texture format documentation from the [Dolphin Emulator](https://github.com/dolphin-emu/dolphin) project
