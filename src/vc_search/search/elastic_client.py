import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ..query_evaluation.relevance_model import RelevanceModel

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
        self.relevance_model = None

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
            "трейдинг, торговля, инвестиции, биржа",
            "сбермобайл, сбер мобайл, сбербанк мобайл",
            "литрес, книги, литература, читалка",
            "vr шлем, vr гарнитура, виртуальная реальность",
            "зумеры, поколение z, молодёжь",
            "эмпатия, сочувствие, понимание",
            "sim карта, симка, сим-карта",
            "роуминг, заграница, зарубежье",
            "ии, искусственный интеллект, ai",
            "нейросети, нейронные сети, neural networks",
            "дизайн, оформление, интерфейс",
            "интерьер, внутреннее убранство, обстановка",
            "бизнес, компания, предприятие, стартап",
            "зарплата, оплата труда, доход, заработок",
            "ожидания, требования, запросы",
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
                            "fuzzy": {
                                "type": "text",
                                "analyzer": "russian_text",
                                "search_analyzer": "russian_text",
                            },
                        },
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "russian_text",
                        "fields": {
                            "english": {"type": "text", "analyzer": "english_text"},
                            "fuzzy": {
                                "type": "text",
                                "analyzer": "russian_text",
                                "search_analyzer": "russian_text",
                            },
                        },
                    },
                    "author": {
                        "type": "text",
                        "analyzer": "russian_text",
                        "fields": {
                            "fuzzy": {
                                "type": "text",
                                "analyzer": "russian_text",
                                "search_analyzer": "russian_text",
                            }
                        },
                    },
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
        """
        Основной поиск
        """
        return self.improved_search(query, limit, sections)

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

    def _analyze_query_type(self, query: str) -> Dict[str, Any]:
        """Анализирует тип запроса для выбора стратегии поиска"""
        query_lower = query.lower()

        has_company_pattern = re.search(
            r"\b(сбер|телеграм|valve|apple|meta)\b", query_lower
        )
        has_product_pattern = re.search(
            r"\b(шлем|подписка|карт|видео|интерьер)\b", query_lower
        )
        has_year_pattern = re.search(r"\b(2025|2024|2023)\b", query)

        analysis = {
            "has_company": bool(has_company_pattern),
            "has_product": bool(has_product_pattern),
            "has_year": bool(has_year_pattern),
            "is_complex": len(query.split()) >= 3,
            "specific_patterns": [],
        }

        query_patterns = {
            "трейдинг": {
                "boost_terms": [
                    "трейдинг",
                    "трейдер",
                    "торговля",
                    "биржа",
                    "инвестиции",
                ]
            },
            "сбермобайл": {
                "boost_terms": ["сбермобайл", "сбер мобайл", "сбербанк мобайл"],
                "exact_match": True,
            },
            "литрес": {
                "boost_terms": ["литрес", "книги", "подписка"],
                "exact_match": True,
            },
            "vr шлем": {
                "boost_terms": ["vr", "виртуальная реальность", "шлем", "гарнитура"]
            },
        }

        for pattern, _ in query_patterns.items():
            if pattern in query_lower:
                analysis["specific_patterns"].append(pattern)

        return analysis

    def improved_search(
        self, query: str, limit: int = 10, sections: List[str] = None
    ) -> Dict:
        analysis = self._analyze_query_type(query)

        # Выбираем стратегию поиска на основе анализа
        if analysis["has_company"] and analysis["has_product"]:
            return self._company_product_search(query, limit, sections)
        elif analysis["has_year"]:
            return self._temporal_search(query, limit, sections)
        elif analysis["is_complex"]:
            return self._complex_query_search(query, limit, sections)
        else:
            return self._universal_search(query, limit, sections)

    def _company_product_search(
        self, query: str, limit: int, sections: List[str] = None
    ) -> Dict:
        """Поиск для запросов типа 'компания + продукт'"""
        search_body = {
            "size": limit,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^5", "content^3", "author^1"],
                                "type": "phrase",
                                "boost": 3.0,
                            }
                        },
                        {
                            "bool": {
                                "must": [
                                    {
                                        "multi_match": {
                                            "query": self._extract_key_terms(query),
                                            "fields": ["title^4", "content^2"],
                                            "operator": "and",
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "content^1.5"],
                                "fuzziness": "AUTO",
                                "prefix_length": 1,
                                "boost": 0.5,
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
                    "content": {"fragment_size": 150, "number_of_fragments": 3},
                },
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"published_date": {"order": "desc"}},
            ],
        }

        if sections:
            search_body["query"]["bool"]["filter"] = [{"terms": {"section": sections}}]

        return self._execute_search(search_body)

    def _temporal_search(
        self, query: str, limit: int, sections: List[str] = None
    ) -> Dict:
        """Поиск для запросов с временными метками"""
        year_match = re.search(r"\b(2025|2024|2023)\b", query)
        if not year_match:
            return self._universal_search(query, limit, sections)

        year = year_match.group(1)
        query_without_year = re.sub(r"\b(2025|2024|2023)\b", "", query).strip()

        search_body = {
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query_without_year,
                                "fields": ["title^4", "content^2", "author^1"],
                                "operator": "and",
                                "fuzziness": "AUTO",
                            }
                        }
                    ],
                    "filter": [
                        {
                            "range": {
                                "published_date": {
                                    "gte": f"{year}-01-01",
                                    "lte": f"{year}-12-31",
                                }
                            }
                        }
                    ],
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

        return self._execute_search(search_body)

    def _complex_query_search(
        self, query: str, limit: int, sections: List[str] = None
    ) -> Dict:
        """Поиск для сложных многословных запросов"""
        terms = query.split()

        # Разделяем запрос на основные компоненты
        if len(terms) >= 3:
            # Первые 2-3 слова - ядро запроса
            core_query = " ".join(terms[:3])
            additional_terms = terms[3:] if len(terms) > 3 else []
        else:
            core_query = query
            additional_terms = []

        search_body = {
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": core_query,
                                "fields": ["title^4", "content^2"],
                                "type": "best_fields",
                                "minimum_should_match": "75%",
                                "fuzziness": "1",
                            }
                        }
                    ],
                    "should": [
                        # Буст для точного совпадения всего запроса
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^5", "content^3"],
                                "type": "phrase",
                                "boost": 2.0,
                            }
                        },
                        # Буст для дополнительных терминов с поддержкой опечаток
                        *[
                            {
                                "match": {
                                    "content.fuzzy": {
                                        "query": term,
                                        "boost": 0.5,
                                        "fuzziness": "AUTO",
                                    }
                                }
                            }
                            for term in additional_terms
                        ],
                    ],
                }
            },
            "highlight": {
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"],
                "fields": {
                    "title": {"number_of_fragments": 0},
                    "content": {"fragment_size": 150, "number_of_fragments": 3},
                },
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"published_date": {"order": "desc"}},
            ],
        }

        if sections:
            search_body["query"]["bool"]["filter"] = [{"terms": {"section": sections}}]

        return self._execute_search(search_body)

    def _universal_search(
        self, query: str, limit: int, sections: List[str] = None
    ) -> Dict:
        """Универсальный поиск для всех остальных запросов"""
        terms = query.split()

        # Автоматическое определение уровня fuzzy на основе длины запроса
        if len(terms) == 1:
            fuzziness = "AUTO"
            prefix_length = 1
        elif len(terms) == 2:
            fuzziness = "1"
            prefix_length = 2
        else:
            fuzziness = "1"
            prefix_length = 3

        search_body = {
            "size": limit,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^4", "content^2", "author^1.5"],
                                "type": "best_fields",
                                "fuzziness": fuzziness,
                                "prefix_length": prefix_length,
                                "boost": 2.0,
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title.fuzzy^3",
                                    "content.fuzzy^1.5",
                                    "author.fuzzy^1",
                                ],
                                "type": "best_fields",
                                "boost": 1.0,
                            }
                        },
                        {
                            "match_phrase": {
                                "content": {"query": query, "slop": 3, "boost": 1.5}
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

        return self._execute_search(search_body)

    def _extract_key_terms(self, query: str) -> str:
        """Извлекает ключевые термины из запроса с учетом тематики"""
        stop_words = {
            "в",
            "для",
            "и",
            "или",
            "на",
            "с",
            "по",
            "о",
            "об",
            "как",
            "где",
            "что",
            "почему",
            "зачем",
            "когда",
            "если",
            "то",
            "это",
            "так",
            "же",
            "но",
            "а",
            "да",
            "нет",
        }

        tech_terms = {"crm", "api", "sdk", "ui", "ux", "vr", "ar", "ai", "ml", "iot"}
        crypto_terms = {"btc", "eth", "usdt", "defi", "nft", "dao"}

        terms = []
        for term in query.split():
            term_lower = term.lower()
            if (
                term_lower not in stop_words
                or term_lower in tech_terms
                or term_lower in crypto_terms
                or re.search(r"\d", term)
            ):  # Сохраняем термины с цифрами
                terms.append(term)

        return " ".join(terms)

    def _execute_search(self, search_body: Dict) -> Dict:
        """Выполняет поисковый запрос"""
        try:
            response = self.client.search(index=self.index_name, body=search_body)
            return self._format_search_results(response)
        except Exception as e:
            logger.error(f"Ошибка улучшенного поиска: {e}")
            return {"results": [], "total": 0, "took": 0}

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

    def search_with_relevance_model(
        self,
        query: str,
        limit: int = 10,
        model_path: str = None,
        use_full_content: bool = True,
        threshold: float = 0.3,
    ) -> Dict:
        """
        Поиск с использованием обученной модели релевантности.

        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            model_path: Путь к файлу модели
            use_full_content: Использовать полный контент или только превью
            threshold: Порог вероятности для фильтрации результатов (0.0-1.0)

        Returns:
            Словарь с результатами поиска
        """
        # Сначала выполняем расширенный поиск
        search_results = self.improved_search(
            query,
            limit=limit * 3,  # Берем больше результатов для фильтрации моделью
        )

        if not search_results["results"]:
            return search_results

        # Проверяем, загружена ли модель
        if self.relevance_model is None and model_path:
            self.load_relevance_model(model_path)

        if self.relevance_model is None:
            # Если модель не загружена, возвращаем обычные результаты
            search_results["results"] = search_results["results"][:limit]
            return search_results

        # Получаем полный контент статей для лучшего предсказания
        if use_full_content:
            try:
                # Получаем полный контент для каждой статьи
                article_ids = [result["id"] for result in search_results["results"]]

                # Запрашиваем полные документы из Elasticsearch
                response = self.client.mget(
                    index=self.index_name,
                    body={"ids": article_ids},
                    _source_includes=["content", "title", "author", "section"],
                )

                # Создаем словарь id -> полный контент
                full_contents = {}
                for doc in response["docs"]:
                    if doc["found"]:
                        full_contents[doc["_id"]] = {
                            "content": doc["_source"].get("content", ""),
                            "title": doc["_source"].get("title", ""),
                            "author": doc["_source"].get("author", ""),
                            "section": doc["_source"].get("section", ""),
                        }

                # Обновляем результаты с полным контентом
                for result in search_results["results"]:
                    if result["id"] in full_contents:
                        full_data = full_contents[result["id"]]
                        # Комбинируем метаданные для лучшей оценки
                        result["full_content_for_prediction"] = (
                            f"{full_data.get('title', '')} "
                            f"{full_data.get('content', '')} "
                            f"{full_data.get('author', '')} "
                            f"{full_data.get('section', '')}"
                        )
                    else:
                        # Если не удалось получить полный контент, используем то, что есть
                        result["full_content_for_prediction"] = (
                            f"{result.get('title', '')} "
                            f"{result.get('content_preview', '')} "
                            f"{' '.join(result.get('highlights', []))}"
                        )

            except Exception as e:
                logger.warning(f"⚠️ Не удалось получить полный контент: {e}")
                use_full_content = False

        # Подготавливаем данные для предсказания
        queries = [query] * len(search_results["results"])

        if use_full_content:
            contents = [
                result.get("full_content_for_prediction", "")
                for result in search_results["results"]
            ]
        else:
            contents = [
                f"{result.get('title', '')} "
                f"{result.get('content_preview', '')} "
                f"{' '.join(result.get('highlights', []))}"
                for result in search_results["results"]
            ]

        try:
            # Предсказываем релевантность
            predictions, probabilities = self.relevance_model.predict_batch(
                queries, contents
            )

            # Добавляем предсказания к результатам
            for i, result in enumerate(search_results["results"]):
                result["relevance_prediction"] = int(predictions[i])
                result["relevance_probability"] = float(probabilities[i])
                # Дополнительный вес для объединенного скоринга
                result["combined_score"] = (
                    result["score"] * 0.3  # Вес Elasticsearch score
                    + result["relevance_probability"] * 0.7  # Вес модели
                )

            # Фильтруем по порогу вероятности
            filtered_results = [
                result
                for result in search_results["results"]
                if result["relevance_probability"] >= threshold
            ]

            if not filtered_results:
                logger.info(f"⚠️ Модель не нашла результатов выше порога {threshold}")
                # Возвращаем топ результатов по Elasticsearch score
                filtered_results = sorted(
                    search_results["results"], key=lambda x: x["score"], reverse=True
                )[:limit]
            else:
                # Сортируем по комбинированному скорингу
                filtered_results = sorted(
                    filtered_results, key=lambda x: x["combined_score"], reverse=True
                )[:limit]

            # Обновляем результаты
            search_results["results"] = filtered_results
            search_results["total"] = len(filtered_results)
            search_results["model_threshold"] = threshold
            search_results["model_used"] = True

            # Логирование статистики
            if len(predictions) > 0:
                avg_prob = sum(probabilities) / len(probabilities)
                relevant_count = sum(predictions)
                logger.info(
                    f"✅ Модель оценена: avg_prob={avg_prob:.3f}, "
                    f"relevant={relevant_count}/{len(predictions)}"
                )

        except Exception as e:
            logger.error(f"❌ Ошибка предсказания модели: {e}", exc_info=True)
            # В случае ошибки возвращаем обычные результаты
            search_results["results"] = search_results["results"][:limit]
            search_results["model_used"] = False

        return search_results

    def load_relevance_model(self, model_path: str):
        """Загрузка модели релевантности"""
        try:
            if not Path(model_path).exists():
                logger.warning(f"⚠️ Файл модели не найден: {model_path}")
                return False

            self.relevance_model = RelevanceModel()
            self.relevance_model.load(model_path)
            logger.info(f"✅ Модель релевантности загружена из {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False
