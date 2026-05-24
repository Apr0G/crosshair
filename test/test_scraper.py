import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scraper import get_demo_urls_from_page

match_id = sys.argv[1] if len(sys.argv) > 1 else "1-ecb64f76-ee62-4b86-bcd9-9968b36fec62"

print(f"testing: {match_id}")
urls = get_demo_urls_from_page(match_id)
print(f"\nfound {len(urls)} demo(s):")
for u in urls:
    print(f"  {u[:120]}")
