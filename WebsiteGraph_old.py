import logging
import requests
import re
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import networkx as nx
from pyvis.network import Network
from typing import Set, Dict, Any, Optional
from requests_cache import CachedSession
from collections import deque
from colorsys import hsv_to_rgb
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import os
import ipynbname

def get_this_ipynb():
    """Gets this ipynb absolute path

    Returns:
        str: path
    """
    try:
        nb_path = ipynbname.path()
        return str(nb_path)
    except:
        try:
            return globals()['__vsc_ipynb_file__']
        except:
            return os.getcwd() + '\\' + 'WebsiteGraphMP.ipynb'
            

def list_files(directory):
    files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            files.append(item)
    return files

def ensure_directory_exists(directory_path):
    # Проверяем, существует ли директория 
    if not os.path.isdir(directory_path):
        # Создаем директорию (включая родительские каталоги при необходимости)
        os.makedirs(directory_path)
    return directory_path

def append_to_file(file_path, text):
    """
    Дописывает строку в конец файла. Если файл не существует - создает его.
    
    :param file_path: Путь к файлу
    :param text: Добавляемая строка
    """
    with open(file_path, 'a') as file:  # Режим 'a' для добавления в конец 
        file.write(text + '\n')  # Добавляем перенос строки 

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("website_graph.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Настройка логгирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("website_graph.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def html_title(html):
    soup = BeautifulSoup(html, 'html.parser')
    container = soup.new_tag('div')
    container.append(soup)
    return container


class WebsiteGraph:
    def __init__(
        self,
        start_url: str,
        max_depth: int = 2,
        domain_filter: Optional[str] = None,
        path_regex: Optional[str] = None,
        node_size: str = "degree",
        layout: Optional[Dict[str, Any]] = None,
        max_links: int = 10,
        expire_after: int = 3000
    ):
        """Класс для построения и визуализации графа веб-сайта.
    
    Args:
        start_url (str): Начальный URL для парсинга.
        max_depth (int, optional): Максимальная глубина обхода. Defaults to 2.
        domain_filter (str, optional): Домен для фильтрации ссылок. Defaults to None.
        path_regex (str, optional): Регулярное выражение для путей. Defaults to None.
        node_size (str, optional): Метод расчета размера узлов. Defaults to "degree".
        layout (Dict[str, Any], optional): Параметры визуализации. Defaults to None.
        max_links (int, optional): Максимум ссылок на страницу. Defaults to 10.
        """
        logger.info(f"Инициализация парсера с параметрами: "
                    f"start_url={start_url}, max_depth={max_depth}, "
                    f"domain_filter={domain_filter}, path_regex={path_regex}, "
                    f"node_size={node_size}, max_links={max_links}")
        
        self.graph = nx.DiGraph()
        self.start_url = self._normalize_url(start_url)
        self.max_depth = max_depth
        self.domain = urlparse(self.start_url).netloc
        self.domain_filter = domain_filter or self.domain
        self.path_regex = re.compile(path_regex) if path_regex else None
        self.node_size = node_size
        self.layout = layout or {"physics": True, "hierarchical": False}
        self.max_links = max_links
        self.expire_after = expire_after
        self.visited = set()
                # Инициализация кэшированной сессии
        self.session = CachedSession(
            cache_name=f'cache/{urlparse(self.start_url).netloc}',  # Отдельный кэш для каждого домена [[2]]
            expire_after=self.expire_after,
            allowable_methods=('GET',)  # Кэшируем только GET-запросы [[7]]
        )
        # Отключаем ненужные проверки для ускорения
        self.session.verify = True  # Используйте с осторожностью! Для HTTPS лучше включить проверку
         
    def _normalize_url(self, url: str) -> str:
        """Нормализует URL, удаляя якори и дублирующие слеши.
        
        Args:
            url (str): Исходный URL.
            
        Returns:
            str: Нормализованный URL.
        """
        parsed = urlparse(url)
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/'),  # Удаляем trailing slash
            '',  # params
            parsed.query,  # Сохраняем query-параметры
            ''   # fragment
        ))
        return normalized

    def _get_article_title(self, url: str) -> str:
        """Упрощенная версия с использованием walrus operator"""
        if (match := re.search(r'/wiki/([^/]+)', url)) and (title := match.group(1)):
            return title.replace('_', ' ')
        return url.split('//')[-1].split('/')[0]  # Альтернатива для не-wiki URL

    def _is_valid_url(self, url: str) -> bool:
        """Проверяет валидность URL согласно фильтрам.
        
        Args:
            url (str): Проверяемый URL.
            
        Returns:
            bool: True, если URL соответствует условиям.
        """
        parsed = urlparse(url)
        is_main_page = parsed.path.lower().endswith('main_page')
        valid = (
            self.domain_filter in parsed.netloc and
            not is_main_page and
            (not self.path_regex or self.path_regex.search(parsed.path))
        )
        if not valid:
            logger.debug(f"URL отклонен: {url} (домен: {parsed.netloc}, путь: {parsed.path})")
        return valid
    
    def _extract_links(self, url: str):
        """Извлекает ссылки со страницы с кэшированием.
        Args:
            url (str): URL страницы для парсинга.
        Returns:
            Tuple[Set[str], Optional[int]]: Множество валидных ссылок и HTTP-статус.
        """
        logger.info(f"Парсинг ссылок с: {url}")
        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()  # Вызывает HTTPError для 4xx/5xx
            soup = BeautifulSoup(response.text, "html.parser")
            links = set()
            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link["href"])
                normalized_url = self._normalize_url(full_url)
                if self._is_valid_url(normalized_url):
                    links.add(normalized_url)
                    if len(links) >= self.max_links:
                        logger.debug(f"Достигнут лимит ссылок ({self.max_links}) для {url}")
                        break
            logger.debug(f"Найдено {len(links)} валидных ссылок на странице")
            return links, response.status_code  # Возвращаем статус успешного ответа

        except requests.exceptions.HTTPError as e:
            # Обработка HTTP-ошибок (4xx/5xx)
            status_code = e.response.status_code if e.response else None
            logger.error(f"HTTP ошибка {status_code} для {url}: {str(e)}", exc_info=True)
            return set(), status_code

        except requests.exceptions.RequestException as e:
            # Обработка сетевых ошибок (timeout, connection error)
            logger.error(f"Сетевая ошибка для {url}: {str(e)}", exc_info=True)
            return set(), None

        except Exception as e:
            # Обработка остальных исключений
            logger.error(f"Непредвиденная ошибка для {url}: {str(e)}", exc_info=True)
            return set(), None

    def _crawl(self, force_reload: bool = False) -> None:
        """Рекурсивно обходит сайт, строя граф ссылок."""
        self.start_crawl_time = time.time()
        if not force_reload and self.graph.nodes:
            logger.info("Используем существующий граф")
            return
        logger.info("Начинаем обход сайта")
        queue = deque([(self.start_url, 0)])
        current_depth = 0  # Начальная глубина
        with tqdm(total=self.max_depth, desc="Глубина обхода") as pbar:
            while queue:
                url, depth = queue.popleft()
                logger.debug(f"Обрабатываем URL: {url} (глубина: {depth}/{self.max_depth})")
                
                # Обновляем прогресс-бар при переходе на новый уровень глубины
                if depth > current_depth:
                    pbar.update(depth - current_depth)
                    current_depth = depth
                    logger.info(f"Переход на глубину: {current_depth}")
                    
                if depth > self.max_depth or url in self.visited:
                    logger.debug(f"Пропуск URL: {url} (посещено: {url in self.visited}, глубина: {depth})")
                    continue
                    
                self.visited.add(url)
                logger.info(f"Добавление узла: {url} (глубина: {depth})")
                
                links, status_code = self._extract_links(url)
                if status_code:  # Если статус получен (успешный запрос)
                    self.graph.add_node(
                        url,
                        title=url,
                        label=self._get_article_title(url),
                        size=self._calculate_node_size(url),
                        status_code=status_code,  # Сохраняем статус
                        depth=depth 
                    )
                else:  # Если произошла ошибка
                    self.graph.add_node(
                        url,
                        title=url,
                        label=self._get_article_title(url),
                        size=self._calculate_node_size(url),
                        status_code=0,  # Или другое значение по умолчанию
                        depth=depth 
                    )
                
                '''links, status = self._extract_links(url)
                self.graph.add_node(
                    url,
                    title=url,
                    label=self._get_article_title(url),
                    size=self._calculate_node_size(url)
                )
                '''
                for link in links:
                    if url != link:
                        logger.debug(f"Добавление связи: {url} -> {link}")
                        #print(f"Добавление связи: {url} -> {link}")
                        self.graph.add_edge(url, link)
                        queue.extend((link, depth+1) for link in links)
        self.end_crawl_time = time.time()

    def _calculate_node_size(self, node: str) -> int:
        """Рассчитывает размер узла на основе входящих связей.
        
        Args:
            node (str): URL узла.
            
        Returns:
            int: Размер узла в пикселях.
        """
        in_degree = dict(self.graph.in_degree()).get(node, 0)
        size = 10 + 3 * in_degree if self.node_size == "degree" else 10
        logger.debug(f"Размер узла {node} рассчитан как {size} (входящих связей: {in_degree})")
        return size
      
    def _get_node_color(self, node: str) -> str:
        """Вычисляет цвет узла от светло-зеленого до рыжего
        Args:
            node (str): URL узла
        Returns:
            str: HEX-код цвета
        """
        if self.max_degree == self.min_degree:
            return "#D4EDD4"  # Светло-зеленый для графа без связей [[2]]
        
        t = (self.graph.in_degree(node) - self.min_degree) / (self.max_degree - self.min_degree)
        
        # Интерполяция от зеленого (h=0.33) к рыжему (h=0.08) [[6]]
        h = 0.33 - 0.25 * t  # 0.33 (зеленый) → 0.08 (рыжий)
        s = 0.8 + 0.2 * t    # Увеличение насыщенности для темных оттенков
        v = 0.9 + 0.1 * t    # Увеличение яркости для светлых участков
        
        r, g, b = hsv_to_rgb(h, s, v)
        return '#{:02x}{:02x}{:02x}'.format(
            int(r*255), int(g*255), int(b*255)
        )

    def visualize(self, rebuild: bool = False, force_reload: bool = False) -> None:
        """Создает HTML-визуализацию графа с использованием pyvis."""
        if rebuild or force_reload or not self.graph.nodes:
            self._crawl(force_reload=force_reload)
        logger.info("Запуск визуализации графа")
        
        
        if not self.graph.nodes:
            logger.warning("Граф пуст - визуализация невозможна")
            return
        
        # Расчет границ градиента
        self.degree_map = dict(self.graph.in_degree())
        self.min_degree = min(self.degree_map.values()) if self.degree_map else 1
        self.max_degree = max(self.degree_map.values()) if self.degree_map else 1
        logger.debug(f"Градиент границы: min={self.min_degree}, max={self.max_degree}")

        net = Network(
            notebook=False,
            directed=True,
            height="800px",
            width="100%",
            cdn_resources="remote"
        )
        
        # Настройки подсветки
        options = {
            "edges": {
                "color": {
                    "color": "#2B7CE9",  # Стандартный цвет
                    "highlight": "#FF0000",  # Цвет подсветки
                    "hover": "#FF0000"       # Цвет при наведении
                },
                "selectionWidth": 3,  # Толщина выделенных рёбер
                "smooth": False
            },
            "interaction": {
                "hoverConnectedEdges": True,
                "selectConnectedEdges": True,  # Автоматический выбор связанных рёбер
                "multiselect": True,
                "navigationButtons": True,
                "keyboard":True,
                "hover": True,
                "click": True,
            },
            "physics": {
                "enabled": True,
                "forceAtlas2Based": {
                    "gravitationalConstant": -200,  # Увеличьте отталкивание (от -50 до -500)
                    "springLength": 500,           # Длина связей между узлами (от 100 до 500)
                    "springConstant": 0.001,        # Жесткость связей (от 0.001 до 0.1)
                    "damping": 0.3,                # Затухание движения (0-1)
                    "avoidOverlap": 1              # Избегать пересечений (0-1)
                },
                "stabilization": {
                    "iterations": 500,             # Итераций для стабилизации
                    "updateInterval": 50
                }
            },
            "nodes": {
                "allow_html": True,  # Включаем поддержку HTML
                "shape": "box",  # Обязательно для кликабельности [[8]]
                "font": {"size": 10},
                "color": {
                    "border": "#2B7CE9",
                    "background": "#97C2FC",
                    "highlight": {
                        "border": "#FF0000",  # Цвет границы узла при выделении
                        "background": "#FFFF00"
                    }
                },
                "chosen": True,
                "style": "cursor: pointer;",
                "shapeProperties": {
                    "allowHtml": True  # Правильный параметр вместо allow_html [[9]]
                    }
            },
        
            "configure": {
                "enabled": False,
                "filter": "nodes,edges",
                "showButton": False
            },
            "version": "9.1.2" 
            }

        net.set_options(json.dumps(options))
        for node in self.graph.nodes:
            
            # Формирование HTML-подсказки
            status_code = self.graph.nodes[node].get('status_code', 0)
            in_degree = self.graph.in_degree(node)
            status_color = "#e6ffe6"  # Зеленый фон по умолчанию
            
            tooltip = (
            f"<div style='padding: 8px; background: {status_color}'>"
            f"<b>URL:</b> {node}<br>"
            f"<b>Status:</b> {status_code}<br>"
            f"<b>In-Degree:</b> {in_degree}<br>"
            f"</div>"
            )
            
            
            title = self._get_article_title(node)
            if status_code and 400 <= int(status_code) < 600:
                color = "#ffcccc"  # Красный фон для ошибок
            else:
                color = self._get_node_color(node)
            color = self._get_node_color(node)  # <- Новое вычисление цвета
            logger.debug(f"Добавление узла в визуализацию: {node} (заголовок: {title})")
            logger.debug(f"Цвет узла {node}: {color} (степень: {self.graph.in_degree(node)})")
            
            # В цикле добавления узлов
            full_url = node if node.startswith(("http://", "https://")) else f"http://{node}"
            escaped_url = full_url.replace("'", "\\'")  # Экранируем одинарные кавычки [[4]]

            
            net.add_node(
                node,
                label=title,
                title=tooltip,
                size=self._calculate_node_size(node),
                color=color,
                url=full_url,
                allow_html=True,
                # Добавляем обработчик клика через JavaScript
                onclick=f"window.open('{escaped_url}', '_blank');",
                shapeProperties={
                "allowHtml": True  # Корректное название опции [[1]]
            },
            )

        for edge in self.graph.edges:
            logger.debug(f"Добавление связи в визуализацию: {edge[0]} -> {edge[1]}")
            net.add_edge(edge[0], edge[1])

        ipynb_dir =  '\\'.join(get_this_ipynb().split('\\')[:-1])
        directory = ensure_directory_exists(ipynb_dir + '\\graphs')
        try:
            graph_num = [int(i.split('.')[0].replace('graph','')) for i in list_files(directory)][-1]+1
        except:
            graph_num = 0
        logger.info("Сохранение графа в HTML-файл")
        file = f"{directory}\\graph{graph_num}.html"
        
        text = f'{file} | max_depth: {self.max_depth} | max_links: {self.max_links} | crawl time: {self.end_crawl_time - self.start_crawl_time}'
        
        net.write_html(file, open_browser=True)
        append_to_file(ipynb_dir+f'\\{self.__name__().split('.')[-1]}.txt',text)
        
        
        logger.info(f"Graph saved as {file} and opened in browser")
        
        
        print(text)
        
        
        # Нужно закрыть сессию)
        self.session.close()
    
    def __name__(self):
        return 'WebsiteGraph'