from dataclasses import dataclass
from typing import List


@dataclass
class ScrapingConfig:
    sections: List[str]
    articles_per_section: int = 500
    delay: float = 0.1
    headless: bool = True
    batch_size: int = 10
    max_workers: int = 4
    timeout: int = 0.1


DEFAULT_CONFIG = ScrapingConfig(
    sections=[
        "services",
        "ai",
        "crypto",
        "telegram",
        "tech",
        "dev",
        "future",
        "midjourney",
        "chatgpt",
        "links",
    ],
    articles_per_section=500,
    delay=0.1,
    headless=True,
    batch_size=10,
    max_workers=4,
    timeout=0.1,
)
