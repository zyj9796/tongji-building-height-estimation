from __future__ import annotations

import argparse
import base64
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
SVG = f"{{{SVG_NS}}}"
XLINK_HREF = f"{{{XLINK_NS}}}href"
REFERENCE_PATTERN = re.compile(r"(?:url\()?\#([A-Za-z_][\w:.-]*)")
SEMANTIC_ID_PREFIXES = ("projection-", "classification-")

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def element_counts(root: ET.Element) -> Counter:
    return Counter(element.tag.rsplit("}", 1)[-1] for element in root.iter())


def collect_referenced_ids(root: ET.Element) -> set[str]:
    referenced: set[str] = set()
    for element in root.iter():
        for value in element.attrib.values():
            referenced.update(REFERENCE_PATTERN.findall(value))
    return referenced


def remove_unreferenced_ids(root: ET.Element, referenced: set[str]) -> None:
    for element in root.iter():
        identifier = element.get("id")
        if (
            identifier
            and identifier not in referenced
            and not identifier.startswith(SEMANTIC_ID_PREFIXES)
        ):
            del element.attrib["id"]


def merge_sibling_paths(parent: ET.Element, referenced: set[str]) -> int:
    merged = 0
    children = list(parent)
    index = 0
    while index < len(children):
        child = children[index]
        if child.tag != SVG + "path" or child.get("id") in referenced:
            index += 1
            continue
        signature = tuple(sorted((key, value) for key, value in child.attrib.items() if key not in {"d", "id"}))
        paths = [child.get("d", "")]
        following = index + 1
        while following < len(children):
            candidate = children[following]
            candidate_signature = tuple(
                sorted((key, value) for key, value in candidate.attrib.items() if key not in {"d", "id"})
            )
            if (
                candidate.tag != SVG + "path"
                or candidate.get("id") in referenced
                or candidate_signature != signature
            ):
                break
            paths.append(candidate.get("d", ""))
            parent.remove(candidate)
            following += 1
            merged += 1
        if len(paths) > 1:
            child.set("d", " ".join(paths))
        children = list(parent)
        index += 1
    for child in list(parent):
        merged += merge_sibling_paths(child, referenced)
    return merged


def flatten_attribute_free_groups(parent: ET.Element) -> int:
    flattened = 0
    changed = True
    while changed:
        changed = False
        for child in list(parent):
            flattened += flatten_attribute_free_groups(child)
            if child.tag != SVG + "g" or child.attrib:
                continue
            position = list(parent).index(child)
            grandchildren = list(child)
            parent.remove(child)
            for offset, grandchild in enumerate(grandchildren):
                parent.insert(position + offset, grandchild)
            flattened += 1
            changed = True
            break
    return flattened


def recompress_image(element: ET.Element) -> tuple[int, int, str] | None:
    attribute = XLINK_HREF if XLINK_HREF in element.attrib else "href"
    href = element.get(attribute, "")
    if not href.startswith("data:image/") or "," not in href:
        return None
    _, encoded = href.split(",", 1)
    raw = base64.b64decode(encoded)
    if len(raw) < 8_000:
        return None
    image = Image.open(io.BytesIO(raw)).convert("RGBA")
    opaque = image.getchannel("A").getextrema() == (255, 255)
    output = io.BytesIO()
    if opaque and image.width * image.height >= 100_000:
        image.convert("RGB").save(
            output,
            format="JPEG",
            quality=90,
            optimize=True,
            progressive=True,
            subsampling=0,
        )
        mime, method = "image/jpeg", "jpeg_q90"
    else:
        quantized = image.quantize(
            colors=256,
            method=Image.Quantize.FASTOCTREE,
            dither=Image.Dither.NONE,
        )
        quantized.save(output, format="PNG", optimize=True, compress_level=9)
        mime, method = "image/png", "png_palette256"
    optimized = output.getvalue()
    if len(optimized) >= len(raw):
        return len(raw), len(raw), "unchanged"
    element.set(attribute, f"data:{mime};base64,{base64.b64encode(optimized).decode('ascii')}")
    return len(raw), len(optimized), method


def optimize_svg_file(path: Path, recompress_images: bool = True) -> dict:
    path = Path(path)
    before_bytes = path.stat().st_size
    tree = ET.parse(path)
    root = tree.getroot()
    before_counts = element_counts(root)
    for metadata in list(root.findall(SVG + "metadata")):
        root.remove(metadata)
    referenced = collect_referenced_ids(root)
    remove_unreferenced_ids(root, referenced)
    merged_paths = merge_sibling_paths(root, referenced)
    flattened_groups = flatten_attribute_free_groups(root)
    image_changes = []
    if recompress_images:
        for image in root.iter(SVG + "image"):
            change = recompress_image(image)
            if change:
                image_changes.append(
                    {
                        "before_bytes": change[0],
                        "after_bytes": change[1],
                        "method": change[2],
                    }
                )
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True, short_empty_elements=True)
    payload = buffer.getvalue()
    path.write_bytes(payload)
    after_counts = element_counts(root)
    return {
        "file": str(path),
        "before_bytes": before_bytes,
        "after_bytes": len(payload),
        "reduction_fraction": 1.0 - len(payload) / before_bytes,
        "before_dom_elements": int(sum(before_counts.values())),
        "after_dom_elements": int(sum(after_counts.values())),
        "merged_paths": int(merged_paths),
        "flattened_groups": int(flattened_groups),
        "image_changes": image_changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Matplotlib SVG while preserving editable text.")
    parser.add_argument("svg", type=Path, nargs="+")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--preserve-images",
        action="store_true",
        help="Keep embedded raster bytes unchanged while optimizing SVG structure.",
    )
    args = parser.parse_args()
    reports = [
        optimize_svg_file(
            path.resolve(),
            recompress_images=not args.preserve_images,
        )
        for path in args.svg
    ]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
