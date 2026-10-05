"""Generic decoder/differ for a length-prefixed binary save format.

Format handled (structurally, no game-specific names hardcoded):
  - little-endian int32 header words,
  - u8-length-prefixed ASCII strings,
  - a u32 field-count followed by a name table of u8-length-prefixed strings,
  - then an opaque value payload (scalars + embedded typed blobs).

Usage:
  python sigsim_save.py parse SAVEFILE
  python sigsim_save.py diff FILE_A FILE_B

Platform notes: cross-platform (Windows + Linux), stdlib only.
Never crashes on unknown layouts: all reads are bounds-checked and
wrapped in try/except; failures degrade to warnings.
"""
import argparse
import struct
import sys
from pathlib import Path

MAX_STR_LEN = 256       # sanity cap for a u8-length-prefixed string
MAX_FIELDS = 100000     # sanity cap for field-count
MAX_REGIONS = 1 << 31   # sanity cap for payload scans


def read_u8_string(data: bytes, off: int):
    """Read one u8-length-prefixed ASCII string at off. Returns (str, next_off)."""
    if off < 0 or off >= len(data):
        raise ValueError("offset %d out of range (size %d)" % (off, len(data)))
    ln = data[off]
    if ln > MAX_STR_LEN:
        raise ValueError("implausible string length %d at offset %d" % (ln, off))
    end = off + 1 + ln
    if end > len(data):
        raise ValueError("string at %d overruns file (need %d, have %d)"
                         % (off, end, len(data)))
    raw = data[off + 1:end]
    return raw.decode("ascii", "replace"), end


def try_read_u32_le(data: bytes, off: int):
    if off < 0 or off + 4 > len(data):
        raise ValueError("u32 read at %d overruns file" % off)
    return struct.unpack_from("<I", data, off)[0], off + 4


def try_read_i32_le(data: bytes, off: int):
    if off < 0 or off + 4 > len(data):
        raise ValueError("i32 read at %d overruns file" % off)
    return struct.unpack_from("<i", data, off)[0], off + 4


def _looks_like_name(s: str):
    # Field names are short printable identifiers; binary garbage decoded
    # as "strings" typically contains control chars or replacement chars.
    return (1 <= len(s) <= MAX_STR_LEN and s.isprintable()
            and "\ufffd" not in s)


def plausible_name_table(data: bytes, off: int):
    """Try to parse a name table (u32 count + count u8-strings) at off.

    Returns (names, next_off) or raises ValueError.
    """
    count, cur = try_read_u32_le(data, off)
    if count == 0 or count > MAX_FIELDS:
        raise ValueError("implausible field count %d at %d" % (count, off))
    names = []
    for _ in range(count):
        s, cur = read_u8_string(data, cur)
        if not _looks_like_name(s):
            raise ValueError("non-name entry %r at %d" % (s, cur))
        names.append(s)
    return names, cur


def find_name_table(data: bytes):
    """Scan for the most plausible name-table offset.

    Strategy: every offset where a u32 count is followed by that many
    consecutive valid u8-strings is a candidate; pick the one with the
    most names (ties -> lowest offset). Returns (off, names, values_off)
    or (None, [], None) if nothing found.
    """
    best = None
    # Only scan the first 64 KiB for the table start (header lives early).
    limit = min(len(data), 65536)
    off = 0
    while off < limit:
        try:
            names, nxt = plausible_name_table(data, off)
            # Require at least 2 names to avoid false positives on scalars.
            if len(names) >= 2 and (best is None or len(names) > len(best[1])):
                best = (off, names, nxt)
                if len(names) > 5000:
                    break  # good enough; tables are rarely huge
        except (ValueError, struct.error):
            pass
        off += 1
    if best is None:
        return None, [], None
    return best


def decode_file(path: Path):
    """Best-effort structural decode. Returns dict with header/names/payload."""
    data = path.read_bytes()
    info = {"path": str(path), "size": len(data), "data": data,
            "header_ints": [], "early_strings": [],
            "table_off": None, "names": [], "values_off": None}
    # Raw header: first up-to-5 little-endian int32s (bounded).
    for i in range(5):
        try:
            v, _ = try_read_i32_le(data, i * 4)
            info["header_ints"].append((i * 4, v))
        except (ValueError, struct.error):
            break
    # Early strings: try a couple of u8-strings right after the int header.
    # Purely informational; failures are ignored.
    off = len(info["header_ints"]) * 4
    for _ in range(4):
        try:
            s, off = read_u8_string(data, off)
            info["early_strings"].append(s)
            # Some layouts put a u32 + padding between string groups; probe it.
            try:
                _cnt, _nxt = try_read_u32_le(data, off)
            except (ValueError, struct.error):
                pass
        except (ValueError, struct.error):
            # Try skipping one alignment/pad byte and continue once.
            try:
                s, off = read_u8_string(data, off + 1)
                info["early_strings"].append(s)
            except (ValueError, struct.error):
                break
            break
    # Name table search (generic, no hardcoded offsets or names).
    try:
        toff, names, voff = find_name_table(data)
    except Exception:
        toff, names, voff = None, [], None
    info["table_off"], info["names"], info["values_off"] = toff, names, voff
    return info


def summarize_scalars(data: bytes, values_off, max_slots=16):
    """Typed guesses for the first bytes of the value payload.

    Shows each 4-byte slot as int32/uint32/float32 without assuming layout.
    """
    lines = []
    if values_off is None or values_off >= len(data):
        return ["<no value payload decoded>"]
    n = min(max_slots * 4, len(data) - values_off)
    for i in range(0, n, 4):
        chunk = data[values_off + i:values_off + i + 4]
        if len(chunk) < 4:
            break
        i32 = struct.unpack("<i", chunk)[0]
        u32 = struct.unpack("<I", chunk)[0]
        f32 = struct.unpack("<f", chunk)[0]
        lines.append("  +%-4d i32=%-12d u32=%-12d f32=%-14.6g hex=%s"
                     % (i, i32, u32, f32, chunk.hex()))
    return lines


def summarize_runs(data: bytes, values_off, max_bytes=256):
    """Run-length summary of the payload prefix (highlights scalar region)."""
    if values_off is None:
        return []
    region = data[values_off:values_off + max_bytes]
    lines = []
    i = 0
    while i < len(region):
        v = region[i]
        j = i
        while j < len(region) and region[j] == v:
            j += 1
        run = j - i
        if v != 0 or run > 1:
            lines.append("  +%-4d 0x%02x (%3d) x%d" % (i, v, v, run))
        i = j
    return lines


def cmd_parse(args):
    try:
        info = decode_file(args.savefile)
    except OSError as e:
        print("error: cannot read %s: %s" % (args.savefile, e), file=sys.stderr)
        return 1
    data = info["data"]
    print("=" * 70)
    print("FILE:", info["path"])
    print("SIZE:", info["size"])
    print("=" * 70)
    print("\n[RAW HEADER 0..32]")
    print(" ".join("%02x" % c for c in data[:32]) or "<empty>")
    print("\n[HEADER INT32s (little-endian)]")
    if info["header_ints"]:
        for off, v in info["header_ints"]:
            print("  int32 @%-3d = %-12d (0x%08x)" % (off, v, v & 0xFFFFFFFF))
    else:
        print("  <unreadable>")
    print("\n[EARLY STRINGS (best effort)]")
    if info["early_strings"]:
        for s in info["early_strings"]:
            print("  %r" % s[:120])
    else:
        print("  <none decoded>")
    print("\n[NAME TABLE]")
    if info["names"]:
        print("  count=%d table@%s values@%s" % (
            len(info["names"]), info["table_off"], info["values_off"]))
        for k, nm in enumerate(info["names"][:50]):
            print("  [%4d] %r" % (k, nm[:100]))
        if len(info["names"]) > 50:
            print("  ... (%d more)" % (len(info["names"]) - 50))
        seen = {}
        for nm in info["names"]:
            seen[nm] = seen.get(nm, 0) + 1
        dupes = {k: v for k, v in seen.items() if v > 1}
        print("  duplicates:", dupes if dupes else "none",
              "| unique:", len(seen))
    else:
        print("  <no plausible name table found>")
    print("\n[VALUE PAYLOAD]")
    if info["values_off"] is not None:
        rest_len = len(data) - info["values_off"]
        print("  starts @%d, length %d" % (info["values_off"], rest_len))
        print("  first 160 bytes:")
        rest = data[info["values_off"]:info["values_off"] + 160]
        for r in range(0, len(rest), 16):
            print("   +%-4d  %s" % (r, " ".join("%02x" % c for c in rest[r:r + 16])))
        print("  typed guesses (first slots):")
        for line in summarize_scalars(data, info["values_off"]):
            print(line)
        print("  run-length view (first %d bytes):" % min(256, rest_len))
        runs = summarize_runs(data, info["values_off"])
        for line in runs[:30]:
            print(line)
        if not runs:
            print("  <all zeros or empty>")
    else:
        print("  <unknown>")
    return 0


def cmd_diff(args):
    try:
        a = decode_file(args.file_a)
        b = decode_file(args.file_b)
    except OSError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    na, nb = a["names"], b["names"]
    da, db = a["data"], b["data"]
    print("%-24s size=%-8d fields=%-6d values@%s" %
          (args.file_a.name, len(da), len(na), a["values_off"]))
    print("%-24s size=%-8d fields=%-6d values@%s" %
          (args.file_b.name, len(db), len(nb), b["values_off"]))
    seta, setb = set(na), set(nb)
    print("\nfield tables identical (ordered):", na == b["names"] if na and nb else na == nb)
    print("field multisets identical:", sorted(na) == sorted(nb))
    only_a = [x for x in na if x not in setb]
    only_b = [x for x in nb if x not in seta]
    print("added in %s (%d): %s" % (args.file_a.name, len(only_a), only_a[:50] or "none"))
    print("only in %s (%d): %s" % (args.file_b.name, len(only_b), only_b[:50] or "none"))
    # Positional changes (same index, different name).
    changed = [(i, x, y) for i, (x, y) in enumerate(zip(na, nb)) if x != y]
    print("positionally changed: %d" % len(changed))
    for i, x, y in changed[:20]:
        print("  [%d] %r -> %r" % (i, x, y))
    n = min(len(da), len(db))
    first = next((i for i in range(n) if da[i] != db[i]), None)
    print("\nfirst differing byte:", first)
    if first is not None:
        lo = max(0, first - 16)
        print(" %s: %s" % (args.file_a.name, " ".join("%02x" % c for c in da[lo:first + 16])))
        print(" %s: %s" % (args.file_b.name, " ".join("%02x" % c for c in db[lo:first + 16])))
    else:
        print("(common prefix identical; sizes: %d vs %d)" % (len(da), len(db)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generic structural decoder/differ "
                                 "for length-prefixed binary save files.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse", help="decode and print header/fields/payload summary")
    p.add_argument("savefile", type=Path, help="save file to decode")
    p.set_defaults(func=cmd_parse)
    d = sub.add_parser("diff", help="compare field tables of two save files")
    d.add_argument("file_a", type=Path, help="first save file")
    d.add_argument("file_b", type=Path, help="second save file")
    d.set_defaults(func=cmd_diff)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
