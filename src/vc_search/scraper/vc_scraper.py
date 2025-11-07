import time
import re
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
import requests
from bs4 import BeautifulSoup

from ..models.article import Article


class VCScraper:
    BASE_URL = "https://vc.ru"
    SECTIONS = [
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
    ]

    def __init__(self, delay: float = 0.1, headless: bool = True):
        self.delay = delay
        self.headless = headless
        self.driver = None
        self._setup_driver()

    def _setup_driver(self):
        try:
            options = Options()

            if self.headless:
                options.add_argument("--headless")

            options.set_preference("dom.webnotifications.enabled", False)
            options.set_preference("media.volume_scale", "0.0")
            options.set_preference(
                "general.useragent.override",
                "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
            )

            service = Service(GeckoDriverManager().install())
            self.driver = webdriver.Firefox(service=service, options=options)
            self.driver.implicitly_wait(0.1)

        except Exception as e:
            print(f"Ошибка запуска Firefox: {e}")
            raise

    def _clean_url(self, url: str) -> str:
        return url.split("#")[0].split("?")[0]

    def _is_article_url(self, url: str, section: str) -> bool:
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) >= 2 and path_parts[0] == section:
            if re.match(r"^\d+", path_parts[1]):
                return True
        return False

    def _extract_article_urls_from_current_page(self, section: str) -> Set[str]:
        urls: Set[str] = set()

        link_selectors = [
            f"a[href*='/{section}/']",
            "a[class*='content-feed']",
            "a[class*='feed__link']",
            "a[class*='content-link']",
        ]

        for selector in link_selectors:
            try:
                links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for link in links:
                    try:
                        href = link.get_attribute("href")
                        if href:
                            clean_url = self._clean_url(href)
                            if self._is_article_url(clean_url, section):
                                urls.add(clean_url)
                    except:
                        continue
            except:
                continue

        return urls

    def scroll_and_collect_urls(
        self, section: str, max_articles: int = 500
    ) -> List[str]:
        section_url = f"{self.BASE_URL}/{section}"
        print(f"Загружаем раздел: {section_url}")

        try:
            self.driver.get(section_url)

            urls: Set[str] = set()
            scroll_attempt = 0
            max_scroll_attempts = 50
            no_new_urls_count = 0

            while (
                len(urls) < max_articles
                and scroll_attempt < max_scroll_attempts
                and no_new_urls_count < 3
            ):
                current_urls = self._extract_article_urls_from_current_page(section)
                new_urls = current_urls - urls

                if new_urls:
                    urls.update(new_urls)
                    no_new_urls_count = 0
                    print(f"Найдено {len(new_urls)} новых статей, всего: {len(urls)}")
                else:
                    no_new_urls_count += 1
                    print(f"Нет новых статей (попытка {no_new_urls_count}/3)")

                if len(urls) < max_articles:
                    self.driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight);"
                    )
                    time.sleep(0.2)

                scroll_attempt += 1

                if scroll_attempt % 5 == 0:
                    print(f"Прокрутка {scroll_attempt}, собрано: {len(urls)} статей")

            print(f"Завершено прокруток: {scroll_attempt}, всего статей: {len(urls)}")
            return list(urls)[:max_articles]

        except Exception as e:
            print(f"Ошибка при загрузке раздела {section}: {e}")
            return []

    def get_article_urls_from_section(
        self, section: str, max_articles: int = 500
    ) -> List[str]:
        return self.scroll_and_collect_urls(section, max_articles)

    def _parse_published_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Парсит дату публикации из тега time"""
        try:
            # Ищем тег time с атрибутом datetime
            time_tag = soup.find("time")
            if time_tag and time_tag.has_attr("datetime"):
                datetime_str = time_tag["datetime"]
                # Парсим ISO формат даты
                return datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))

            # Альтернативные селекторы для даты
            date_selectors = [
                ".content-header__date",
                ".content-header__time",
                ".time",
                ".date",
                '*[class*="time"]',
                '*[class*="date"]',
            ]

            for selector in date_selectors:
                date_element = soup.select_one(selector)
                if date_element:
                    # Пытаемся найти time внутри элемента
                    time_inner = date_element.find("time")
                    if time_inner and time_inner.has_attr("datetime"):
                        datetime_str = time_inner["datetime"]
                        return datetime.fromisoformat(
                            datetime_str.replace("Z", "+00:00")
                        )

        except Exception as e:
            print(f"Ошибка парсинга даты: {e}")

        return None

    def _extract_article_content(self, article_tag) -> str:
        """Извлекает весь текстовый контент из статьи"""
        content_parts = []

        # Основные текстовые элементы для парсинга
        text_selectors = [
            "p",  # параграфы
            "li",  # элементы списка
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",  # заголовки
            "blockquote",  # цитаты
            "figcaption",  # подписи к изображениям
            ".block-text",  # блоки текста
            ".block-list",  # списки
        ]

        # Собираем текст из всех элементов
        for selector in text_selectors:
            elements = article_tag.find_all(selector)
            for element in elements:
                text = element.get_text().strip()

                # Фильтруем рекламу и короткие/неинформативные тексты
                if (
                    len(text) > 10
                    and not re.search(r"(реклама|ads?|sponsored|promoted)", text, re.I)
                    and not re.search(r"^http", text)
                    and not re.search(r"^#\w+$", text)
                ):  # исключаем хештеги без контекста
                    content_parts.append(text)

        # Также собираем текст из всех div с текстовым контентом
        text_divs = article_tag.find_all(
            "div", class_=re.compile(r"text|content|block", re.I)
        )
        for div in text_divs:
            # Проверяем, что это действительно текстовый блок, а не контейнер
            if div.find(["p", "li", "h2", "h3", "h4"]):
                continue  # пропускаем, если внутри уже есть текстовые элементы

            text = div.get_text().strip()
            if (
                len(text) > 20
                and not re.search(r"(реклама|ads?|sponsored)", text, re.I)
                and not re.search(r"^http", text)
            ):
                content_parts.append(text)

        # Убираем дубликаты и объединяем
        unique_content = []
        seen_texts = set()

        for text in content_parts:
            # Нормализуем текст для сравнения (убираем лишние пробелы)
            normalized = re.sub(r"\s+", " ", text).strip()
            if normalized and normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_content.append(text)

        return "\n".join(unique_content) if unique_content else "Нет контента"

    def parse_article(self, url: str, section: str) -> Optional[Article]:
        try:
            time.sleep(self.delay)

            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                },
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Ищем основной контент статьи
            article_tag = soup.find("article") or soup.find(
                "div", class_=re.compile(r"content|article|post", re.I)
            )

            if not article_tag:
                return None

            # Извлекаем заголовок
            title = "Без заголовка"
            title_tag = article_tag.find("h1") or soup.find("h1")
            if title_tag:
                title_text = title_tag.get_text().strip()
                if title_text:
                    title = title_text

            # Извлекаем весь контент статьи
            content = self._extract_article_content(article_tag)

            # Извлекаем автора
            author = "Неизвестный автор"
            author_selectors = [".user__name", ".author__name", '*[class*="author"]']

            for selector in author_selectors:
                author_tag = soup.select_one(selector)
                if author_tag:
                    author_text = author_tag.get_text().strip()
                    if author_text:
                        author = author_text
                        break

            # Парсим дату публикации
            published_date = self._parse_published_date(soup)

            return Article(
                url=url,
                title=title,
                content=content,
                author=author,
                section=section,
                published_date=published_date,
            )

        except Exception as e:
            print(f"Ошибка парсинга статьи {url}: {e}")
            return None

    def close(self):
        if self.driver:
            self.driver.quit()
