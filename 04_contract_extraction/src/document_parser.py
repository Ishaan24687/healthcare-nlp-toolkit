"""
Parse contract text into structured sections using regex and heuristics.
Identifies headers, section numbers, paragraphs, and list items.
In production at Lantern we used Azure Document Intelligence for this — here
I'm replicating the core logic in pure Python to show the approach.
"""
# TODO: add table detection — pricing schedules are often in tabular format
# and regex won't cut it; need something like camelot or tabula

import re
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Section:
    section_id: str
    title: str
    level: int  # 1 = top section, 2 = subsection
    text: str
    start_pos: int
    end_pos: int
    children: list


@dataclass
class ParsedDocument:
    title: str
    preamble: str
    sections: list[Section]
    raw_text: str


SECTION_HEADER_PATTERN = re.compile(
    r'^(?:SECTION\s+)?(\d+)\.\s+([A-Z][A-Z\s,/&]+)$',
    re.MULTILINE
)

SUBSECTION_PATTERN = re.compile(
    r'^(\d+\.\d+)\s+(.+?)\.?\s+(.+)',
    re.MULTILINE
)

LIST_ITEM_PATTERN = re.compile(
    r'^\s*\(([a-z])\)\s+(.+)',
    re.MULTILINE
)

TITLE_PATTERN = re.compile(
    r'^([A-Z][A-Z\s]+(?:AGREEMENT|CONTRACT|ARRANGEMENT|DUA))$',
    re.MULTILINE
)


def parse_contract(contract_text: str) -> ParsedDocument:
    lines = contract_text.strip().split('\n')

    title = _extract_title(contract_text)
    preamble = _extract_preamble(contract_text)
    sections = _extract_sections(contract_text)

    return ParsedDocument(
        title=title,
        preamble=preamble,
        sections=sections,
        raw_text=contract_text,
    )


def _extract_title(contract_text: str) -> str:
    match = TITLE_PATTERN.search(contract_text[:500])
    if match:
        return match.group(1).strip()

    first_lines = contract_text.strip().split('\n')[:5]
    for line in first_lines:
        line = line.strip()
        if line and line.isupper() and len(line) > 10:
            return line
    return "Untitled Contract"


def _extract_preamble(contract_text: str) -> str:
    first_section = SECTION_HEADER_PATTERN.search(contract_text)
    if first_section:
        preamble_text = contract_text[:first_section.start()].strip()
    else:
        # no section headers found — treat first paragraph as preamble
        paragraphs = contract_text.split('\n\n')
        preamble_text = paragraphs[0].strip() if paragraphs else ""

    # clean up the preamble
    lines = preamble_text.split('\n')
    # skip the title line(s)
    cleaned_lines = []
    past_title = False
    for line in lines:
        stripped = line.strip()
        if not past_title and stripped.isupper() and len(stripped) > 5:
            continue
        if stripped:
            past_title = True
        cleaned_lines.append(stripped)

    return '\n'.join(cleaned_lines).strip()


def _extract_sections(contract_text: str) -> list[Section]:
    section_matches = list(SECTION_HEADER_PATTERN.finditer(contract_text))

    if not section_matches:
        return [Section(
            section_id="1",
            title="Full Document",
            level=1,
            text=contract_text,
            start_pos=0,
            end_pos=len(contract_text),
            children=[],
        )]

    sections = []
    for i, match in enumerate(section_matches):
        section_id = match.group(1)
        section_title = match.group(2).strip()
        start_pos = match.start()

        if i + 1 < len(section_matches):
            end_pos = section_matches[i + 1].start()
        else:
            end_pos = len(contract_text)

        section_text = contract_text[match.end():end_pos].strip()

        subsections = _extract_subsections(section_text, section_id)

        sections.append(Section(
            section_id=section_id,
            title=section_title,
            level=1,
            text=section_text,
            start_pos=start_pos,
            end_pos=end_pos,
            children=subsections,
        ))

    return sections


def _extract_subsections(section_text: str, parent_id: str) -> list[Section]:
    subsections = []
    subsection_matches = list(SUBSECTION_PATTERN.finditer(section_text))

    for i, match in enumerate(subsection_matches):
        sub_id = match.group(1)
        sub_title = match.group(2).strip()
        sub_content = match.group(3).strip()

        if i + 1 < len(subsection_matches):
            end_pos = subsection_matches[i + 1].start()
            remaining_text = section_text[match.end():end_pos].strip()
        else:
            remaining_text = section_text[match.end():].strip()

        full_text = sub_content
        if remaining_text:
            full_text += '\n' + remaining_text

        list_items = _extract_list_items(full_text)

        subsections.append(Section(
            section_id=sub_id,
            title=sub_title,
            level=2,
            text=full_text,
            start_pos=match.start(),
            end_pos=match.end() + len(remaining_text),
            children=list_items,
        ))

    return subsections


def _extract_list_items(text: str) -> list:
    items = []
    for match in LIST_ITEM_PATTERN.finditer(text):
        items.append({
            "marker": match.group(1),
            "text": match.group(2).strip(),
        })
    return items


def document_to_dict(doc: ParsedDocument) -> dict:
    def section_to_dict(s: Section) -> dict:
        return {
            "section_id": s.section_id,
            "title": s.title,
            "level": s.level,
            "text": s.text[:500] + "..." if len(s.text) > 500 else s.text,
            "n_children": len(s.children),
            "children": [section_to_dict(c) if isinstance(c, Section) else c for c in s.children],
        }

    return {
        "title": doc.title,
        "preamble": doc.preamble[:300] + "..." if len(doc.preamble) > 300 else doc.preamble,
        "n_sections": len(doc.sections),
        "sections": [section_to_dict(s) for s in doc.sections],
    }


def parse_all_contracts(data_path: str, output_dir: str = "outputs"):
    with open(data_path, "r") as f:
        contracts = json.load(f)

    print("=" * 60)
    print("Contract Document Parser")
    print("=" * 60)

    parsed_results = []
    for contract in contracts:
        parsed = parse_contract(contract["text"])
        result = document_to_dict(parsed)
        result["contract_id"] = contract["id"]
        parsed_results.append(result)

        print(f"\n{contract['id']}: {parsed.title}")
        print(f"  Preamble: {parsed.preamble[:100]}...")
        print(f"  Sections: {len(parsed.sections)}")
        for section in parsed.sections:
            print(f"    {section.section_id}. {section.title} ({len(section.children)} subsections)")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "parsed_contracts.json")
    with open(output_path, "w") as f:
        json.dump(parsed_results, f, indent=2)
    print(f"\nParsed results saved to {output_path}")

    return parsed_results


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "sample_contracts.json")
    output_dir = os.path.join(base_dir, "outputs")

    if not os.path.exists(data_path):
        print("Contract data not found. Generating samples first...")
        from sample_contracts import save_contracts
        save_contracts(os.path.join(base_dir, "data"))

    parse_all_contracts(data_path, output_dir)
