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

        synonyms_list = [
            "ai, искусственный интеллект, искуственный интелект, ИИ, AI",
            "ml, машинное обучение, machine learning",
            "vr, виртуальная реальность, virtual reality",
            "ar, дополненная реальность, augmented reality",
            "crypto, криптовалюта, cryptocurrency, крипта",
            "bitcoin, биткоин, btc, биток",
            "ethereum, эфириум, eth",
            "telegram, телеграм, тг",
            "startup, стартап, старт ап",
            "app, приложение, application",
            "it, информационные технологии, айти",
            "ui, юи, пользовательский интерфейс",
            "ux, юх, пользовательский опыт",
            "api, апи, интерфейс программирования приложений",
            "бизнес, бизнес, бинес",
            "компания, компания, кампания",
        ]

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
                        "russian_synonyms": {
                            "type": "synonym",
                            "synonyms": synonyms_list,
                        },
                    },
                    "analyzer": {
                        "russian_text": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "russian_stop",
                                "russian_stemmer",
                                "russian_synonyms",
                            ],
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

    def index_articles_from_json(self, json_file: str = "data/articles.json") -> Dict:
        json_path = Path(json_file)
        if not json_path.exists():
            raise ValueError(f"Файл {json_file} не существует")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                articles_data = json.load(f)

            logger.info(f"Загружено {len(articles_data)} статей из {json_file}")

            def articles_generator():
                for article_data in articles_data:
                    try:
                        doc_id = article_data["url"].split("/")[-1].split("?")[0]
                        if not doc_id or len(doc_id) > 512:
                            doc_id = hash(article_data["url"]) % 10**8

                        article_data["word_count"] = len(
                            article_data.get("content", "").split()
                        )
                        article_data["scraped_at"] = datetime.now().isoformat()

                        # Обрабатываем дату
                        if article_data.get("published_date"):
                            try:
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
                        logger.error(
                            f"Ошибка обработки статьи {article_data.get('url', 'unknown')}: {e}"
                        )
                        continue

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
            return {"success": success, "errors": errors, "total": len(articles_data)}

        except Exception as e:
            logger.error(f"Ошибка чтения файла {json_file}: {e}")
            return {"success": 0, "errors": 1, "total": 0}

    def search(self, query: str, limit: int = 10, sections: List[str] = None) -> Dict:
        """Поиск по статьям с учетом русского языка и синонимов"""
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

    def search_with_fuzzy(
        self,
        query: str,
        limit: int = 10,
        sections: List[str] = None,
        fuzziness: str = "AUTO",
        prefix_length: int = 2,
    ) -> Dict:
        """Поиск с поддержкой опечаток через fuzzy matching"""

        # Разбиваем запрос на отдельные слова для fuzzy поиска
        query_terms = query.split()
        fuzzy_queries = []

        for term in query_terms:
            if len(term) > 3:  # Применяем fuzzy только к словам длиннее 3 символов
                fuzzy_queries.append(
                    {
                        "multi_match": {
                            "query": term,
                            "fields": ["title^3", "content^2", "author^1.5"],
                            "fuzziness": fuzziness,
                            "prefix_length": prefix_length,
                        }
                    }
                )
            else:
                # Для коротких слов используем точное совпадение
                fuzzy_queries.append(
                    {
                        "multi_match": {
                            "query": term,
                            "fields": ["title^3", "content^2", "author^1.5"],
                        }
                    }
                )

        search_body = {
            "size": limit,
            "query": {
                "bool": {
                    "should": [
                        # Оригинальный запрос (точное совпадение)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "content^2", "author^1.5"],
                                "type": "best_fields",
                            }
                        },
                        # Fuzzy запрос по отдельным словам
                        {"bool": {"must": fuzzy_queries}},
                        # Match phrase с некоторой свободой
                        {
                            "match_phrase": {
                                "content": {"query": query, "slop": 3, "boost": 0.5}
                            }
                        },
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
            return self._format_search_results(response)
        except Exception as e:
            logger.error(f"Ошибка fuzzy поиска: {e}")
            return {"results": [], "total": 0, "took": 0}

    def smart_search(self, query: str, limit: int = 10) -> Dict:
        """Умный поиск с автоматическим определением необходимости fuzzy"""

        # Для коротких запросов используем fuzzy, для длинных - комбинированный подход
        query_terms = query.split()

        if len(query_terms) == 1 and len(query) > 5:
            # Одно слово длиннее 5 символов - используем fuzzy
            return self.search_with_fuzzy(query, limit, fuzziness="AUTO")
        elif len(query_terms) > 1:
            # Многословный запрос - комбинируем подходы
            return self.search_with_fuzzy(
                query, limit, fuzziness="1"
            )  # Более строгий fuzzy
        else:
            # Короткие запросы - обычный поиск
            return self.search(query, limit)

    def _format_search_results(self, response: Dict) -> Dict:
        """Форматирует результаты поиска"""
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
