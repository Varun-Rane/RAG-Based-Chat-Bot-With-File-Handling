# import re 

# text = """My name is Varun Rane and my eamil address is varunrane81@gmail.com
#     you can also contact me on vs7003075@gmail.com
# """
# result = re.findall(r'\S+@\S', text)
# print(result)

# Extraction of Data from Website using docling (Not working for some reason)

# from docling.document_converter import DocumentConverter

# converter = DocumentConverter()

# url = "https://www.geeksforgeeks.org/artificial-intelligence/chunking-strategies/"

# result = converter.convert(url)

# text = result.document.export_to_markdown()

# print(text[:2000])

# Extraction of Data from Website using requests and BeautifulSoup
import requests
from bs4 import BeautifulSoup

url = "https://www.geeksforgeeks.org/artificial-intelligence/chunking-strategies/"

# Fetch webpage
response = requests.get(url)

# Parse HTML
soup = BeautifulSoup(response.text, "html.parser")

# Extract all paragraphs
paragraphs = soup.find_all("p")

# Print paragraph text
for para in paragraphs[:10]:
    print(para.text)


