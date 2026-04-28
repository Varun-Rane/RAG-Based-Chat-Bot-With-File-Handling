from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls

converter = DocumentConverter()

# PDF Extraction
result = converter.convert("https://arxiv.org/pdf/2408.09869")
document = result.document
markdown = document.export_to_markdown()

print("PDF Extracted Successfully\n")
print(markdown[:2000])   # first 2000 chars only


# Scraping Multiple Pages using Sitemap
sitemap_urls = get_sitemap_urls(
    "https://www.freecodecamp.org/news/sitemap.xml",
    max_urls=3
)

print("\nURLs found:", len(sitemap_urls))
print(sitemap_urls)

docs = []

if sitemap_urls:
    for result in converter.convert_all(sitemap_urls):
        md = result.document.export_to_markdown()
        docs.append(md)

print("\nDocs extracted:", len(docs))

for i, doc in enumerate(docs, start=1):
    print(f"\n--- Document {i} ---\n")
    print(doc[:1500])   