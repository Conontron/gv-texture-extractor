# Texture Extractor

# This is an experimental tool which is not yet completed. It's currently only for testing purposes. 

A Python command-line tool for locating and extracting textures from supported Wii GX and Nintendo Switch BNTX containers. It can process individual files, recursively process folders, or scan binary files for embedded textures.

Extracted textures are written as PNG files. A JSON manifest is created for each processed input file.

## Requirements

Install the Python dependencies:

```shell
pip install -r requirements.txt
```

The script requires Python, NumPy, and Pillow.

## Usage

```shell
python gv_tex_extract.py <input> [output] [options]
```

`<input>` may be a file or a directory. In directory mode, the tool searches recursively for files named `TEX` by default. Use `--all-files` to process files regardless of name or extension. Input files do not need a particular filename extension.

| Option | Description |
| --- | --- |
| `-l`, `--list` | List detected textures without extracting them. |
| `-v`, `--verbose` | Print details for each extracted texture. |
| `--palettes unique\|first\|all` | Select palette variants to export for indexed textures. The default is `unique`. |
| `--mips` | Export all mipmap levels. |
| `--all-files` | In directory mode, process every file rather than only files named `TEX`. |
| `--scan` | Scan unsupported containers for embedded WST or BNTX textures. |

Examples:

```shell
# Extract a single file.
python gv_tex_extract.py input_file output_directory

# List textures without writing images.
python gv_tex_extract.py input_file --list

# Scan every file in a directory for embedded textures.
python gv_tex_extract.py input_directory output_directory --all-files --scan
```

## Output

The extractor writes one PNG per texture. Additional suffixes identify palette variants (`_pN`), mipmap levels (`_mipN`), and BNTX layers (`_layerN`).

For each input file containing textures, the output directory also receives `_manifest.json`. It records the discovered texture metadata, extraction results, and any per-texture errors.

## Supported containers

| Container | Support |
| --- | --- |
| `WST` | GX texture data |
| `ArcL` | Nested texture entries |
| `WPT` | Texture entry container |
| `CMP` | Texture entry container |
| `Menu` | Texture entry container |
| `SARC` | Archive container |
| `BNTX` | Nintendo Switch texture container |
| Offset table of WST entries | Texture table |

Unknown containers can be searched with `--scan`.

## Supported texture formats

GX texture decoding includes CMPR, C4, C8, C14X2, I4, I8, IA4, IA8, RGB565, RGB5A3, and RGBA8. Indexed textures support IA8, RGB565, and RGB5A3 palettes.

BNTX decoding includes R8, RG8, RGBA4, RGB565, RGBA8, BGRA8, BC1, BC2, BC3, BC4, BC5, BC6H, and BC7. Compressed BNTX formats are decoded through Pillow when supported by the installed version.

## Notes

The tool validates texture headers and skips entries with unsupported formats or inconsistent sizes. Use `--list` to inspect a file before extraction, and `--scan` when textures may be embedded in an otherwise unsupported binary format.

## Credits

- [Dolphin Emulator](https://github.com/dolphin-emu/dolphin) for GX texture-format reference material.
- [Ryujinx](https://github.com/Ryujinx/Ryujinx) and Switch Toolbox for Tegra block-linear layout reference material.
