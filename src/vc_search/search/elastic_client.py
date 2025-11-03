import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import logging

logger = logging.getLogger(__name__)


class VCElasticSearch:
    def __init__(self, host: str = "localhost", port: int = 9200):
        self.client = Elasticsearch(
            [f"http://{host}:{port}"],
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
        self.index_name = "vc-ru-articles"

    def setup_index(self):
        """Создает индекс с настройками для русского языка"""
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Индекс {self.index_name} уже существует")
            return True

        index_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "filter": {
                        "russian_stop": {"type": "stop", "stopwords": "_russian_"},
                        "russian_stemmer": {"type": "stemmer", "language": "russian"},
                        "english_stop": {"type": "stop", "stopwords": "_english_"},
                        "english_stemmer": {"type": "stemmer", "language": "english"},
                    },
                    "analyzer": {
                        "russian_text": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "russian_stop", "russian_stemmer"],
                        },
                        "english_text": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "english_stop", "english_stemmer"],
                        },
                    },
                },
            },
            "mappings": {
                "properties": {
                    "url": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "russian_text",
                        "fields": {
                            "english": {"type": "text", "analyzer": "english_text"},
                            "keyword": {"type": "keyword"},
                        },
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "russian_text",
                        "fields": {
                            "english": {"type": "text", "analyzer": "english_text"}
                        },
                    },
                    "author": {"type": "text", "analyzer": "russian_text"},
                    "section": {"type": "keyword"},
                    "published_date": {"type": "date"},
                    "word_count": {"type": "integer"},
                    "scraped_at": {"type": "date"},
                }
            },
        }

        try:
            self.client.indices.create(index=self.index_name, body=index_body)
            logger.info(f"Создан индекс {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Ошибка создания индекса: {e}")
            return False

    def index_articles_from_json(self, json_dir: str = "data/articles") -> Dict:
        """Индексирует статьи из JSON файлов"""
        json_path = Path(json_dir)
        if not json_path.exists():
            raise ValueError(f"Директория {json_dir} не существует")

        def articles_generator():
            for json_file in json_path.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        article_data = json.load(f)

                    # Создаем ID из URL
                    doc_id = article_data["url"].split("/")[-1].split("?")[0]
                    if not doc_id or len(doc_id) > 512:
                        doc_id = hash(article_data["url"]) % 10**8

                    # Добавляем дополнительные поля
                    article_data["word_count"] = len(
                        article_data.get("content", "").split()
                    )
                    article_data["scraped_at"] = datetime.now().isoformat()

                    # Обрабатываем дату
                    if article_data.get("published_date"):
                        try:
                            # Приводим дату к правильному формату
                            date_str = article_data["published_date"]
                            if date_str.endswith("Z"):
                                date_str = date_str.replace("Z", "+00:00")
                            article_data["published_date"] = date_str
                        except:
                            article_data["published_date"] = None

                    yield {
                        "_index": self.index_name,
                        "_id": str(doc_id),
                        "_source": article_data,
                    }

                except Exception as e:
                    logger.error(f"Ошибка обработки файла {json_file}: {e}")
                    continue

        try:
            success, errors = bulk(
                self.client,
                articles_generator(),
                stats_only=True,
                chunk_size=100,
                max_retries=2,
            )

            # Принудительно обновляем индекс
            self.client.indices.refresh(index=self.index_name)

            logger.info(f"Успешно проиндексировано: {success}, ошибок: {errors}")
            return {"success": success, "errors": errors}

        except Exception as e:
            logger.error(f"Ошибка bulk индексации: {e}")
            return {"success": 0, "errors": 1}

    def search(self, query: str, limit: int = 10, sections: List[str] = None) -> Dict:
        """Поиск по статьям с учетом русского языка"""
        search_body = {
            "size": limit,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "content^2", "author^1.5"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        },
                        {"match_phrase": {"content": {"query": query, "slop": 2}}},
                    ]
                }
            },
            "highlight": {
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"],
                "fields": {
                    "title": {"number_of_fragments": 0},
                    "content": {"fragment_size": 150, "number_of_fragments": 2},
                },
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"published_date": {"order": "desc"}},
            ],
        }

        if sections:
            search_body["query"]["bool"]["filter"] = [{"terms": {"section": sections}}]

        try:
            response = self.client.search(index=self.index_name, body=search_body)

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                highlight = hit.get("highlight", {})

                all_highlights = []
                if "title" in highlight:
                    all_highlights.extend(highlight["title"])
                if "content" in highlight:
                    all_highlights.extend(highlight["content"])

                results.append(
                    {
                        "id": hit["_id"],
                        "url": source["url"],
                        "title": source.get("title", ""),
                        "content_preview": source.get("content", "")[:200] + "...",
                        "author": source.get("author", ""),
                        "section": source.get("section", ""),
                        "published_date": source.get("published_date"),
                        "word_count": source.get("word_count", 0),
                        "score": hit["_score"],
                        "highlights": all_highlights[:3],
                    }
                )

            return {
                "results": results,
                "total": response["hits"]["total"]["value"],
                "took": response["took"],
            }

        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return {"results": [], "total": 0, "took": 0}

    def get_index_stats(self) -> Dict:
        """Статистика индекса"""
        try:
            stats = self.client.indices.stats(index=self.index_name)
            count = self.client.count(index=self.index_name)

            aggs = {"sections": {"terms": {"field": "section", "size": 20}}}

            agg_response = self.client.search(
                index=self.index_name, body={"size": 0, "aggs": aggs}
            )

            sections_dist = {
                bucket["key"]: bucket["doc_count"]
                for bucket in agg_response["aggregations"]["sections"]["buckets"]
            }

            return {
                "doc_count": count["count"],
                "size_bytes": stats["indices"][self.index_name]["total"]["store"][
                    "size_in_bytes"
                ],
                "sections": sections_dist,
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}

    def health_check(self) -> bool:
        """Проверка здоровья Elasticsearch"""
        try:
            return self.client.ping()
        except:
            return False

    def delete_index(self):
        """Удаляет индекс (для тестирования)"""
        try:
            self.client.indices.delete(index=self.index_name)
            logger.info(f"Индекс {self.index_name} удален")
        except Exception as e:
            logger.error(f"Ошибка удаления индекса: {e}")
