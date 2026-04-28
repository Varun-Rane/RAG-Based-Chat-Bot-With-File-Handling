import xml.etree.ElementTree as ET
from typing import List, Optional, Set
import requests


def _parse_sitemap_xml(
    sitemap_url: str,
    seen_sitemaps: Set[str],
    limit: Optional[int] = None
) -> List[str]:

    response = requests.get(
        sitemap_url,
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    if response.status_code == 404:
        return []

    response.raise_for_status()

    root = ET.fromstring(response.content)
    root_tag = root.tag.lower()

    namespaces = (
        {"ns": root.tag.split("}")[0].strip("{")}
        if "}" in root.tag
        else None
    )

    # If sitemap index
    if "sitemapindex" in root_tag:

        child_sitemaps = (
            [elem.text for elem in root.findall(".//ns:loc", namespaces)]
            if namespaces
            else [elem.text for elem in root.findall(".//loc")]
        )

        urls = []

        for child in child_sitemaps:
            if child and child not in seen_sitemaps:

                seen_sitemaps.add(child)

                child_urls = _parse_sitemap_xml(
                    child,
                    seen_sitemaps,
                    limit
                )

                urls.extend(child_urls)

                if limit and len(urls) >= limit:
                    return urls[:limit]

        return urls[:limit] if limit else urls

    # Normal sitemap
    url_nodes = (
        root.findall(".//ns:loc", namespaces)
        if namespaces
        else root.findall(".//loc")
    )

    urls = [elem.text for elem in url_nodes if elem.text]

    return urls[:limit] if limit else urls


def get_sitemap_urls(
    sitemap_url: str,
    max_urls: Optional[int] = None
) -> List[str]:

    try:
        urls = _parse_sitemap_xml(
            sitemap_url,
            {sitemap_url},
            max_urls
        )

        return list(dict.fromkeys(urls))

    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch sitemap: {e}")

    except ET.ParseError as e:
        raise ValueError(f"Failed to parse sitemap XML: {e}")

    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")


if __name__ == "__main__":
    urls = get_sitemap_urls(
        "https://www.freecodecamp.org/news/sitemap.xml",
        max_urls=5
    )

    print(urls)