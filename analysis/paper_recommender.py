from __future__ import annotations

from dataclasses import dataclass
from typing import List
import xml.etree.ElementTree as ET

import requests

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecommendedPaper:
    title: str
    authors: str
    year: int | None
    abstract: str
    url: str


class PaperRecommender:
    BASE_URL = "http://export.arxiv.org/api/query"

    def recommend(
            self,
            query: str,
            limit: int = 5,
    ) -> List[RecommendedPaper]:

        params = {
            "search_query": f"all:{query}",
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=20,
        )

        response.raise_for_status()

        # arXiv returns XML (Atom format), so we parse it using ET
        root = ET.fromstring(response.content)

        # Atom XML Namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []

        for entry in root.findall("atom:entry", ns):
            # Extract Title
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""

            # Extract Authors
            author_elems = entry.findall("atom:author/atom:name", ns)
            authors = ", ".join(a.text.strip() for a in author_elems if a.text)

            # Extract Year from published date (e.g., '2023-05-12T18:00:00Z')
            published_elem = entry.find("atom:published", ns)
            year = None
            if published_elem is not None and published_elem.text:
                year = int(published_elem.text[:4])

            # Extract Abstract (Summary)
            summary_elem = entry.find("atom:summary", ns)
            abstract = summary_elem.text.strip().replace("\n",
                                                         " ") if summary_elem is not None and summary_elem.text else ""

            # Extract URL (id parameter in arXiv entry contains the canonical paper URL)
            id_elem = entry.find("atom:id", ns)
            url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

            papers.append(
                RecommendedPaper(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    url=url,
                )
            )

        logger.info(f"Found {len(papers)} recommended papers from arXiv")

        return papers