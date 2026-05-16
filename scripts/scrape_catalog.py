"""
SHL Product Catalog Scraper
============================
Scrapes Individual Test Solutions from the SHL product catalog.
Excludes Pre-packaged Job Solutions (type=2).

Catalog URL: https://www.shl.com/solutions/products/product-catalog/
Individual Tests: ?start=0&type=1 through ?start=372&type=1 (32 pages, 12 per page)

For each assessment, visits the detail page to extract:
- Name, URL, Description, Test Types, Duration, Job Levels,
  Remote Testing, Adaptive/IRT, Languages

Output: data/catalog.json
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
import sys

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"
ITEMS_PER_PAGE = 12
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "catalog.json")

# Request headers to mimic browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Rate limiting
DELAY_BETWEEN_PAGES = 1.5  # seconds between listing page requests
DELAY_BETWEEN_DETAILS = 1.0  # seconds between detail page requests


def get_page(url, retries=3):
    """Fetch a page with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"  [RETRY {attempt + 1}/{retries}] Error fetching {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  [FAILED] Could not fetch {url}")
                return None


def scrape_listing_page(start, page_type=1):
    """
    Scrape a single listing page to get assessment names and URLs.
    
    The listing page has a table with columns:
    - Individual Test Solutions (name as link)
    - Remote Testing (green dot if yes)
    - Adaptive/IRT (green dot if yes)
    
    Returns list of dicts with: name, url, remote_testing, adaptive
    """
    url = f"{CATALOG_URL}?start={start}&type={page_type}"
    print(f"[PAGE] Scraping listing: {url}")
    
    html = get_page(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, "html.parser")
    assessments = []
    
    # Find the table/section with Individual Test Solutions
    # The catalog page renders assessment rows in a table structure
    # Each row has: name (link), remote testing (dot), adaptive (dot)
    
    # Look for all links that point to /products/product-catalog/view/
    links = soup.find_all("a", href=re.compile(r"/products/product-catalog/view/"))
    
    seen_urls = set()
    for link in links:
        href = link.get("href", "")
        name = link.get_text(strip=True)
        
        if not name or not href:
            continue
            
        # Make URL absolute
        if href.startswith("/"):
            full_url = BASE_URL + href
        else:
            full_url = href
        
        # Deduplicate (links can appear multiple times in HTML)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        
        # Get the parent row to check for remote testing and adaptive dots
        # The table row contains the assessment info
        parent_row = link.find_parent("tr") or link.find_parent("div", class_=re.compile(r"row|product|catalog"))
        
        remote_testing = False
        adaptive = False
        
        if parent_row:
            # Green dots are typically represented as specific elements
            # Check all cells/columns after the name
            cells = parent_row.find_all("td") or parent_row.find_all("div")
            if len(cells) >= 2:
                # Second column: Remote Testing
                remote_cell = cells[1] if len(cells) > 1 else None
                if remote_cell:
                    # Green dot can be a span with specific class, or a bullet character
                    remote_testing = bool(
                        remote_cell.find("span", class_=re.compile(r"dot|green|check|yes|catalogue__circle", re.I))
                        or "●" in remote_cell.get_text()
                        or remote_cell.find("span", attrs={"style": re.compile(r"green", re.I)})
                    )
                # Third column: Adaptive/IRT  
                adaptive_cell = cells[2] if len(cells) > 2 else None
                if adaptive_cell:
                    adaptive = bool(
                        adaptive_cell.find("span", class_=re.compile(r"dot|green|check|yes|catalogue__circle", re.I))
                        or "●" in adaptive_cell.get_text()
                        or adaptive_cell.find("span", attrs={"style": re.compile(r"green", re.I)})
                    )
        
        assessments.append({
            "name": name,
            "url": full_url,
            "remote_testing": remote_testing,
            "adaptive": adaptive,
        })
    
    print(f"  Found {len(assessments)} assessments on this page")
    return assessments


def scrape_detail_page(url):
    """
    Scrape an individual assessment's detail page to get full details.
    
    Detail page structure (from HTML analysis):
    #### Description
    The test measures...
    
    #### Job levels
    Professional Individual Contributor, Mid-Professional,
    
    #### Languages
    English (USA),
    
    #### Assessment length
    Approximate Completion Time in minutes = 30
    Test Type: K
    Remote Testing: [Yes/empty]
    """
    html = get_page(url)
    if not html:
        return {}
    
    soup = BeautifulSoup(html, "html.parser")
    details = {}
    
    # Get the main content area (after breadcrumbs)
    # The detail page has sections marked with h4 headers
    
    # Method 1: Look for h4 elements with specific text
    all_text = soup.get_text(separator="\n")
    
    # Extract Description
    desc_match = re.search(
        r"Description\s*\n+(.*?)(?=\n\s*(?:Job levels|Languages|Assessment length|Downloads|$))",
        all_text,
        re.DOTALL | re.IGNORECASE
    )
    if desc_match:
        details["description"] = desc_match.group(1).strip()
    
    # Extract Job Levels
    job_match = re.search(
        r"Job levels?\s*\n+(.*?)(?=\n\s*(?:Languages|Assessment length|Downloads|$))",
        all_text,
        re.DOTALL | re.IGNORECASE
    )
    if job_match:
        raw_levels = job_match.group(1).strip().rstrip(",")
        # Parse comma-separated job levels
        details["job_levels"] = [
            level.strip() for level in raw_levels.split(",") 
            if level.strip()
        ]
    
    # Extract Languages
    lang_match = re.search(
        r"Languages?\s*\n+(.*?)(?=\n\s*(?:Assessment length|Downloads|$))",
        all_text,
        re.DOTALL | re.IGNORECASE
    )
    if lang_match:
        raw_langs = lang_match.group(1).strip().rstrip(",")
        details["languages"] = [
            lang.strip() for lang in raw_langs.split(",")
            if lang.strip()
        ]
    
    # Extract Assessment Length / Duration
    length_match = re.search(
        r"(?:Assessment length|Completion Time)\s*\n*(.*?)(?=\n\s*(?:Downloads|Speak to|Book a|Back to|$))",
        all_text,
        re.DOTALL | re.IGNORECASE
    )
    if length_match:
        length_text = length_match.group(1).strip()
        
        # Extract duration in minutes
        duration_match = re.search(r"(\d+)\s*(?:minutes|mins)", length_text, re.IGNORECASE)
        if duration_match:
            details["duration"] = f"{duration_match.group(1)} minutes"
        else:
            # Try "Completion Time in minutes = XX"
            time_match = re.search(r"minutes\s*=\s*(\d+)", length_text, re.IGNORECASE)
            if time_match:
                details["duration"] = f"{time_match.group(1)} minutes"
        
        # Extract Test Type
        type_match = re.search(r"Test Type:\s*([A-Z,\s]+)", length_text, re.IGNORECASE)
        if type_match:
            raw_types = type_match.group(1).strip().rstrip(",")
            details["test_types"] = [
                t.strip() for t in raw_types.split(",")
                if t.strip() and t.strip() in "ABCDEKPS"
            ]
        
        # Extract Remote Testing from detail page
        remote_match = re.search(r"Remote Testing:\s*(Yes|No)?", length_text, re.IGNORECASE)
        if remote_match:
            details["remote_testing_detail"] = remote_match.group(1) == "Yes" if remote_match.group(1) else None
    
    # Also try to find Adaptive/IRT info
    adaptive_match = re.search(r"Adaptive\s*/?\s*IRT:\s*(Yes|No)?", all_text, re.IGNORECASE)
    if adaptive_match and adaptive_match.group(1):
        details["adaptive_detail"] = adaptive_match.group(1).lower() == "yes"
    
    # Try OG description as fallback
    if "description" not in details:
        og_desc = soup.find("meta", property="og:description")
        if og_desc:
            content = og_desc.get("content", "")
            # OG description format: "Name: description text"
            if ":" in content:
                details["description"] = content.split(":", 1)[1].strip()
            else:
                details["description"] = content.strip()
    
    return details


def scrape_all_listing_pages():
    """
    Scrape all listing pages for Individual Test Solutions (type=1).
    The catalog has 32 pages, 12 items per page.
    """
    all_assessments = []
    seen_urls = set()
    
    # First, determine the total number of pages
    # We know from analysis: 32 pages for type=1 (start=0 to start=372)
    # But let's discover dynamically by checking pagination
    
    start = 0
    page_num = 1
    max_pages = 35  # Safety limit
    
    while page_num <= max_pages:
        print(f"\n{'='*60}")
        print(f"Scraping listing page {page_num} (start={start})")
        print(f"{'='*60}")
        
        assessments = scrape_listing_page(start, page_type=1)
        
        if not assessments:
            print(f"No assessments found on page {page_num}. Stopping.")
            break
        
        # Deduplicate against already seen URLs
        new_count = 0
        for assessment in assessments:
            if assessment["url"] not in seen_urls:
                seen_urls.add(assessment["url"])
                all_assessments.append(assessment)
                new_count += 1
        
        print(f"  Added {new_count} new assessments (total: {len(all_assessments)})")
        
        # If we got fewer than ITEMS_PER_PAGE, we're on the last page
        if len(assessments) < ITEMS_PER_PAGE:
            print(f"Last page reached (got {len(assessments)} < {ITEMS_PER_PAGE})")
            break
        
        start += ITEMS_PER_PAGE
        page_num += 1
        time.sleep(DELAY_BETWEEN_PAGES)
    
    return all_assessments


def enrich_with_details(assessments):
    """
    Visit each assessment's detail page to get full details.
    Enriches the assessment dict with: description, test_types, duration, 
    job_levels, languages.
    """
    total = len(assessments)
    
    for i, assessment in enumerate(assessments):
        print(f"\n[{i+1}/{total}] Fetching details: {assessment['name']}")
        print(f"  URL: {assessment['url']}")
        
        details = scrape_detail_page(assessment["url"])
        
        # Merge details into assessment
        if details.get("description"):
            assessment["description"] = details["description"]
        else:
            assessment["description"] = ""
            
        if details.get("test_types"):
            assessment["test_types"] = details["test_types"]
        else:
            assessment["test_types"] = []
            
        if details.get("duration"):
            assessment["duration"] = details["duration"]
        else:
            assessment["duration"] = ""
            
        if details.get("job_levels"):
            assessment["job_levels"] = details["job_levels"]
        else:
            assessment["job_levels"] = []
            
        if details.get("languages"):
            assessment["languages"] = details["languages"]
        else:
            assessment["languages"] = []
        
        # Use detail page remote_testing if available (more reliable)
        if details.get("remote_testing_detail") is not None:
            assessment["remote_testing"] = details["remote_testing_detail"]
        
        # Use detail page adaptive if available
        if details.get("adaptive_detail") is not None:
            assessment["adaptive"] = details["adaptive_detail"]
        
        # Print summary
        print(f"  Description: {assessment['description'][:80]}..." if assessment['description'] else "  Description: N/A")
        print(f"  Test Types: {assessment['test_types']}")
        print(f"  Duration: {assessment['duration']}")
        print(f"  Job Levels: {assessment['job_levels']}")
        print(f"  Remote: {assessment['remote_testing']}, Adaptive: {assessment['adaptive']}")
        
        time.sleep(DELAY_BETWEEN_DETAILS)
    
    return assessments


def validate_catalog(assessments):
    """Validate the scraped catalog data."""
    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}")
    
    total = len(assessments)
    print(f"Total assessments: {total}")
    
    # Check for required fields
    missing_desc = sum(1 for a in assessments if not a.get("description"))
    missing_types = sum(1 for a in assessments if not a.get("test_types"))
    missing_duration = sum(1 for a in assessments if not a.get("duration"))
    missing_levels = sum(1 for a in assessments if not a.get("job_levels"))
    
    print(f"Missing description: {missing_desc}/{total}")
    print(f"Missing test_types: {missing_types}/{total}")
    print(f"Missing duration: {missing_duration}/{total}")
    print(f"Missing job_levels: {missing_levels}/{total}")
    
    # Check URL validity
    invalid_urls = [a for a in assessments if not a["url"].startswith("https://www.shl.com/")]
    print(f"Invalid URLs: {len(invalid_urls)}/{total}")
    
    # Check for duplicates
    urls = [a["url"] for a in assessments]
    duplicates = len(urls) - len(set(urls))
    print(f"Duplicate URLs: {duplicates}")
    
    # Test type distribution
    type_counts = {}
    for a in assessments:
        for t in a.get("test_types", []):
            type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nTest Type Distribution:")
    for t in sorted(type_counts.keys()):
        type_labels = {
            "A": "Ability & Aptitude",
            "B": "Biodata & Situational Judgement", 
            "C": "Competencies",
            "D": "Development & 360",
            "E": "Assessment Exercises",
            "K": "Knowledge & Skills",
            "P": "Personality & Behavior",
            "S": "Simulations",
        }
        label = type_labels.get(t, "Unknown")
        print(f"  {t} ({label}): {type_counts[t]}")
    
    # Remote testing & adaptive counts
    remote_count = sum(1 for a in assessments if a.get("remote_testing"))
    adaptive_count = sum(1 for a in assessments if a.get("adaptive"))
    print(f"\nRemote Testing: {remote_count}/{total}")
    print(f"Adaptive/IRT: {adaptive_count}/{total}")
    
    return len(invalid_urls) == 0 and duplicates == 0


def save_catalog(assessments, output_file):
    """Save catalog to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(output_file)
    print(f"\nSaved {len(assessments)} assessments to {output_file}")
    print(f"File size: {file_size / 1024:.1f} KB")


def main():
    print("=" * 60)
    print("SHL PRODUCT CATALOG SCRAPER")
    print("Scope: Individual Test Solutions ONLY (type=1)")
    print("Excludes: Pre-packaged Job Solutions (type=2)")
    print("=" * 60)
    
    # Step 1: Scrape all listing pages
    print("\n[STEP 1/4] Scraping listing pages...")
    assessments = scrape_all_listing_pages()
    
    if not assessments:
        print("ERROR: No assessments found! Check if the website is accessible.")
        sys.exit(1)
    
    print(f"\n[STEP 1 COMPLETE] Found {len(assessments)} assessments from listing pages")
    
    # Step 2: Enrich with detail page data
    print(f"\n[STEP 2/4] Fetching detail pages for {len(assessments)} assessments...")
    print(f"Estimated time: ~{len(assessments) * DELAY_BETWEEN_DETAILS / 60:.0f} minutes")
    assessments = enrich_with_details(assessments)
    
    print(f"\n[STEP 2 COMPLETE] All detail pages scraped")
    
    # Step 3: Validate
    print(f"\n[STEP 3/4] Validating catalog...")
    is_valid = validate_catalog(assessments)
    
    if not is_valid:
        print("\nWARNING: Validation found issues. Review the report above.")
    else:
        print("\nAll validations passed!")
    
    # Step 4: Save
    print(f"\n[STEP 4/4] Saving catalog...")
    save_catalog(assessments, OUTPUT_FILE)
    
    print(f"\n{'='*60}")
    print("SCRAPING COMPLETE!")
    print(f"Total assessments: {len(assessments)}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
