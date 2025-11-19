"""
Extract structured entities from contract text: dates, monetary amounts,
percentages, party names, SLA metrics, and compliance references.
Returns results as structured JSON — the kind of output you'd feed into
a contract management database.
"""

import re
import json
import os
from datetime import datetime
from typing import Optional

from dateutil import parser as date_parser

# date patterns in contracts
DATE_PATTERNS = [
    re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'),
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),
    re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
    re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'),
]

MONETARY_PATTERN = re.compile(
    r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:per\s+(?:member|employee|patient|prescription|visit|transaction|hour|occurrence|year|month)|'
    r'PMPM|PEPM|PALPM|PBPY|million|billion))?',
    re.IGNORECASE,
)

PERCENTAGE_PATTERN = re.compile(
    r'(?:\d+(?:\.\d+)?)\s*%(?:\s+(?:of|above|below|uptime|availability|accuracy|discount|savings|risk))?',
    re.IGNORECASE,
)

SLA_PATTERNS = [
    re.compile(r'(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:clean\s+)?claims?\s+(?:within|processed)', re.IGNORECASE),
    re.compile(r'(?:system|platform)\s+(?:availability|uptime)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%', re.IGNORECASE),
    re.compile(r'(\d+(?:\.\d+)?)\s*%\s*(?:system|platform)\s+(?:availability|uptime)', re.IGNORECASE),
    re.compile(r'(?:answer|respond)\s+(\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?(?:calls?|inquiries)', re.IGNORECASE),
    re.compile(r'within\s+(\d+)\s+(?:hours?|minutes?|seconds?|business\s+days?|days?)', re.IGNORECASE),
    re.compile(r'(\d+)\s*-?\s*(?:hour|minute|second|day)\s+(?:response|resolution|turnaround)', re.IGNORECASE),
    re.compile(r'(?:accuracy|completeness)\s+(?:rate\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*%', re.IGNORECASE),
]

COMPLIANCE_KEYWORDS = {
    "HIPAA": r'\bHIPAA\b',
    "HITECH": r'\bHITECH\b(?:\s+Act)?',
    "42 CFR Part 2": r'42\s+CFR\s+Part\s+2',
    "42 CFR Part 425": r'42\s+CFR\s+Part\s+425',
    "42 CFR § 423.504": r'42\s+CFR\s+§?\s*423\.504',
    "45 CFR Parts 160/164": r'45\s+CFR\s+Parts?\s+(?:160|164)',
    "45 CFR § 164.408": r'45\s+CFR\s+§?\s*164\.408',
    "45 CFR § 164.514": r'45\s+CFR\s+§?\s*164\.514',
    "21 CFR Part 11": r'21\s+CFR\s+Part\s+11',
    "21 CFR Part 1311": r'21\s+CFR\s+Part\s+1311',
    "Anti-Kickback Statute": r'Anti-?Kickback\s+Statute',
    "Stark Law": r'Stark\s+Law',
    "SOC 2 Type II": r'SOC\s+2\s+Type\s+II',
    "ISO 27001": r'ISO\s+27001',
    "NCQA": r'\bNCQA\b',
    "GCP/ICH E6": r'(?:Good\s+Clinical\s+Practice|ICH\s+E6)',
    "DEA": r'\bDEA\b\s+(?:registration|regulations|compliance)',
    "FDA": r'\bFDA\b',
    "CMS MSSP": r'CMS\s+MSSP',
    "OIG Guidance": r'OIG\s+guidance',
}

PARTY_PATTERNS = [
    re.compile(r'(?:between|by and between)\s+(.+?)\s*(?:\(|,\s*a\s+)', re.IGNORECASE),
    re.compile(r'(?:and|between)\s+(.+?)\s*(?:\(|,\s*a\s+)', re.IGNORECASE),
    re.compile(r'"([^"]+)"\s*\)', re.IGNORECASE),
]


def extract_dates(contract_text: str) -> list[dict]:
    dates = []
    seen_positions = set()

    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(contract_text):
            if match.start() not in seen_positions:
                raw_date = match.group(0)
                try:
                    parsed = date_parser.parse(raw_date)
                    normalized = parsed.strftime("%Y-%m-%d")
                except (ValueError, OverflowError):
                    normalized = raw_date

                # grab surrounding context to understand what the date refers to
                context_start = max(0, match.start() - 60)
                context_end = min(len(contract_text), match.end() + 60)
                context = contract_text[context_start:context_end].strip()
                context = re.sub(r'\s+', ' ', context)

                dates.append({
                    "raw": raw_date,
                    "normalized": normalized,
                    "context": context,
                    "position": match.start(),
                })
                seen_positions.add(match.start())

    return dates


def extract_monetary_amounts(contract_text: str) -> list[dict]:
    amounts = []
    for match in MONETARY_PATTERN.finditer(contract_text):
        raw = match.group(0)

        numeric_str = re.sub(r'[^\d.]', '', raw.split()[0])
        try:
            numeric_value = float(numeric_str)
        except ValueError:
            numeric_value = None

        context_start = max(0, match.start() - 60)
        context_end = min(len(contract_text), match.end() + 60)
        context = re.sub(r'\s+', ' ', contract_text[context_start:context_end].strip())

        amounts.append({
            "raw": raw,
            "value": numeric_value,
            "context": context,
            "position": match.start(),
        })

    return amounts


def extract_percentages(contract_text: str) -> list[dict]:
    percentages = []
    for match in PERCENTAGE_PATTERN.finditer(contract_text):
        raw = match.group(0)
        numeric_str = re.search(r'[\d.]+', raw)
        value = float(numeric_str.group()) if numeric_str else None

        context_start = max(0, match.start() - 60)
        context_end = min(len(contract_text), match.end() + 60)
        context = re.sub(r'\s+', ' ', contract_text[context_start:context_end].strip())

        percentages.append({
            "raw": raw,
            "value": value,
            "context": context,
            "position": match.start(),
        })

    return percentages


def extract_sla_metrics(contract_text: str) -> list[dict]:
    sla_metrics = []
    for pattern in SLA_PATTERNS:
        for match in pattern.finditer(contract_text):
            context_start = max(0, match.start() - 30)
            context_end = min(len(contract_text), match.end() + 80)
            context = re.sub(r'\s+', ' ', contract_text[context_start:context_end].strip())

            sla_metrics.append({
                "raw_match": match.group(0),
                "metric_value": match.group(1) if match.lastindex else match.group(0),
                "context": context,
                "position": match.start(),
            })

    # deduplicate by position proximity
    if sla_metrics:
        deduped = [sla_metrics[0]]
        for metric in sla_metrics[1:]:
            if all(abs(metric["position"] - d["position"]) > 10 for d in deduped):
                deduped.append(metric)
        sla_metrics = deduped

    return sla_metrics


def extract_compliance_references(contract_text: str) -> list[dict]:
    references = []
    for ref_name, pattern_str in COMPLIANCE_KEYWORDS.items():
        pattern = re.compile(pattern_str, re.IGNORECASE)
        matches = list(pattern.finditer(contract_text))
        if matches:
            references.append({
                "reference": ref_name,
                "count": len(matches),
                "first_position": matches[0].start(),
            })
    return sorted(references, key=lambda x: x["first_position"])


def extract_party_names(contract_text: str) -> list[str]:
    # look for quoted abbreviations like ("PBM") or ("Client")
    abbreviation_pattern = re.compile(r'"([^"]{2,50})"\s*\)')
    parties = set()

    header = contract_text[:1000]

    for match in abbreviation_pattern.finditer(header):
        abbrev = match.group(1)
        # look backwards for the full name
        before_text = contract_text[:match.start()]
        full_name_match = re.search(
            r'([A-Z][A-Za-z\s,]+(?:Inc\.|LLC|Corp\.|P\.A\.|P\.C\.|L\.P\.))',
            before_text[-200:],
        )
        if full_name_match:
            parties.add(full_name_match.group(1).strip())

    return list(parties)


def extract_all_terms(contract_text: str, contract_id: str = "") -> dict:
    return {
        "contract_id": contract_id,
        "dates": extract_dates(contract_text),
        "monetary_amounts": extract_monetary_amounts(contract_text),
        "percentages": extract_percentages(contract_text),
        "sla_metrics": extract_sla_metrics(contract_text),
        "compliance_references": extract_compliance_references(contract_text),
        "party_names": extract_party_names(contract_text),
    }


def process_all_contracts(data_path: str, output_dir: str = "outputs"):
    print("=" * 60)
    print("Contract Term Extraction")
    print("=" * 60)

    with open(data_path, "r") as f:
        contracts = json.load(f)

    all_extractions = []
    for contract in contracts:
        extraction = extract_all_terms(contract["text"], contract["id"])
        all_extractions.append(extraction)

        print(f"\n{contract['id']}: {contract['title']}")
        print(f"  Dates found: {len(extraction['dates'])}")
        if extraction['dates']:
            for d in extraction['dates'][:3]:
                print(f"    {d['normalized']} — {d['context'][:60]}...")
        print(f"  Monetary amounts: {len(extraction['monetary_amounts'])}")
        if extraction['monetary_amounts']:
            for m in extraction['monetary_amounts'][:3]:
                print(f"    {m['raw']}")
        print(f"  Percentages: {len(extraction['percentages'])}")
        print(f"  SLA metrics: {len(extraction['sla_metrics'])}")
        print(f"  Compliance refs: {len(extraction['compliance_references'])}")
        if extraction['compliance_references']:
            refs = [r['reference'] for r in extraction['compliance_references']]
            print(f"    {', '.join(refs)}")
        print(f"  Parties: {extraction['party_names']}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "extracted_terms.json")
    with open(output_path, "w") as f:
        json.dump(all_extractions, f, indent=2)
    print(f"\nExtractions saved to {output_path}")

    total_dates = sum(len(e["dates"]) for e in all_extractions)
    total_amounts = sum(len(e["monetary_amounts"]) for e in all_extractions)
    total_pcts = sum(len(e["percentages"]) for e in all_extractions)
    total_slas = sum(len(e["sla_metrics"]) for e in all_extractions)
    total_compliance = sum(len(e["compliance_references"]) for e in all_extractions)

    print(f"\n{'='*60}")
    print(f"Extraction Summary across {len(contracts)} contracts:")
    print(f"  Total dates:        {total_dates}")
    print(f"  Total amounts:      {total_amounts}")
    print(f"  Total percentages:  {total_pcts}")
    print(f"  Total SLA metrics:  {total_slas}")
    print(f"  Total compliance:   {total_compliance}")

    return all_extractions


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "sample_contracts.json")
    output_dir = os.path.join(base_dir, "outputs")

    if not os.path.exists(data_path):
        print("Contract data not found. Generating samples first...")
        from sample_contracts import save_contracts
        save_contracts(os.path.join(base_dir, "data"))

    process_all_contracts(data_path, output_dir)
