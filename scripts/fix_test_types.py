"""
Fix test_types extraction in catalog.json
==========================================
The original scraper missed test_types because "Test Type:" and the value 
are on separate lines with whitespace. This script re-fetches detail pages
to extract test_types correctly, and also cleans up job_levels that leaked.
"""

import json
import requests
from bs4 import BeautifulSoup
import re
import time
import os

CATALOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "catalog.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

VALID_TEST_TYPES = {"A", "B", "C", "D", "E", "K", "P", "S"}


def extract_test_type_and_remote(html):
    """Extract test type and remote testing from detail page HTML."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    
    result = {"test_types": [], "remote_testing": False, "adaptive": False}
    
    # Test Type is on a line after "Test Type:" with whitespace/newlines between
    # Pattern: Test Type:\n                                    \n\n\nK\n
    type_match = re.search(r'Test Type:\s*\n\s*([A-Z](?:\s*,\s*[A-Z])*)', text)
    if type_match:
        raw = type_match.group(1).strip()
        result["test_types"] = [t.strip() for t in raw.split(",") if t.strip() in VALID_TEST_TYPES]
    
    # Remote Testing: check if there's a green circle/dot after "Remote Testing:"
    # In HTML, it's typically a span element near "Remote Testing:" text
    remote_match = re.search(r'Remote Testing:\s*\n', text)
    if remote_match:
        # Check if there's content (circle/dot) between "Remote Testing:" and next section
        pos = remote_match.end()
        next_100 = text[pos:pos+100].strip()
        # If the next non-whitespace is a visible element (not just whitespace), remote is yes
        result["remote_testing"] = len(next_100) > 0 and not next_100.startswith("Download")
    
    # Also try the HTML approach for remote testing
    # Look for the remote testing section with a green circle span
    remote_spans = soup.find_all("span", class_=re.compile(r"catalogue__circle", re.I))
    if remote_spans:
        result["remote_testing"] = True
    
    # Adaptive/IRT
    adaptive_match = re.search(r'Adaptive\s*/?\s*IRT', text)
    if adaptive_match:
        result["adaptive"] = True
    
    return result


def fix_job_levels(levels):
    """Clean up job levels that have test type data leaked into them."""
    clean_levels = []
    valid_level_prefixes = [
        "Entry", "General", "Graduate", "Mid", "Professional", 
        "Front Line", "Manager", "Supervisor", "Director", "Executive",
        "Senior"
    ]
    
    for level in levels:
        # Check if this is a real job level (not leaked test type data)
        level = level.strip()
        if any(level.startswith(prefix) for prefix in valid_level_prefixes):
            clean_levels.append(level)
        # Skip entries that contain "Test Type:" or other non-level data
    
    return clean_levels


def main():
    print("Loading catalog.json...")
    with open(CATALOG_FILE, "r") as f:
        catalog = json.load(f)
    
    total = len(catalog)
    print(f"Total assessments: {total}")
    
    fixed_count = 0
    job_fixed = 0
    
    for i, assessment in enumerate(catalog):
        # Fix job_levels (remove leaked test type data)
        old_levels = assessment.get("job_levels", [])
        clean_levels = fix_job_levels(old_levels)
        if len(clean_levels) != len(old_levels):
            assessment["job_levels"] = clean_levels
            job_fixed += 1
        
        # Only re-fetch if test_types is empty
        if not assessment.get("test_types"):
            print(f"[{i+1}/{total}] Fixing: {assessment['name']}")
            
            try:
                r = requests.get(assessment["url"], headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    result = extract_test_type_and_remote(r.text)
                    
                    if result["test_types"]:
                        assessment["test_types"] = result["test_types"]
                        fixed_count += 1
                        print(f"  → Test Types: {result['test_types']}")
                    else:
                        print(f"  → No test type found on page")
                    
                    # Also update remote_testing from detail page
                    assessment["remote_testing"] = result["remote_testing"]
                else:
                    print(f"  → HTTP {r.status_code}")
            except Exception as e:
                print(f"  → Error: {e}")
            
            time.sleep(0.5)  # Rate limit
    
    # Save fixed catalog
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    
    # Validation report
    missing_types = sum(1 for a in catalog if not a.get("test_types"))
    type_counts = {}
    for a in catalog:
        for t in a.get("test_types", []):
            type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"FIX COMPLETE")
    print(f"{'='*60}")
    print(f"Test types fixed: {fixed_count}/{total}")
    print(f"Job levels cleaned: {job_fixed}/{total}")
    print(f"Still missing test_types: {missing_types}/{total}")
    print(f"\nTest Type Distribution:")
    for t in sorted(type_counts.keys()):
        labels = {"A": "Ability", "B": "Biodata/SJT", "C": "Competencies", 
                  "D": "Development", "E": "Exercises", "K": "Knowledge",
                  "P": "Personality", "S": "Simulations"}
        print(f"  {t} ({labels.get(t, '?')}): {type_counts[t]}")


if __name__ == "__main__":
    main()
