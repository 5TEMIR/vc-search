from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    url: str
    title: str
    content: str
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    section: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "published_date": self.published_date.isoformat()
            if self.published_date
            else None,
            "author": self.author,
            "section": self.section,
        }
