
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import networkx as nx
from pyvis.network import Network
from requests_cache import CachedSession
from colorsys import hsv_to_rgb
import requests
import numpy as np
import time
import pandas as pd
from statsmodels.api import OLS
from sklearn.preprocessing import PolynomialFeatures
import ipynbname
import os
import pickle

def read_github_txt_file(url, save_locally=False, file_path="file.txt"):
    """
    Скачивает TXT-файл с GitHub и возвращает его содержимое.
    
    Args:
        url (str): Ссылка на RAW-файл (через raw.githubusercontent.com) [[7]][[8]]
        save_locally (bool): Сохранить файл локально (по умолчанию False)
        file_path (str): Путь для сохранения (по умолчанию "file.txt")
    
    Returns:
        str: Содержимое файла
    """
    # Скачивание файла
    response = requests.get(url)
    response.raise_for_status()  # Проверка ошибок 
    
    # Сохранение локально (опционально)
    if save_locally:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Файл сохранен: {file_path}")  # Информация о сохранении
    
    # Возврат содержимого в виде текста
    return response.content.decode("utf-8")  # Декодирование бинарных данных 

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

class WebsiteGraphMP: # MP for MultiProcessing
    def __init__(self,
                 start_url: str = '',
                 max_depth: int = 2,
                 max_links: int = 5,
                 path_regex: str = None,
                 workers: int = 10,
                 expire_after: int = 3000,
                 node_size: str = "degree",
                 layout: dict = None,
                 show_menu : bool = False
                 ):
        self.start_url = self._normalize_url(start_url)
        self.max_depth = max_depth
        self.max_links = max_links
        self.domain = urlparse(self.start_url).netloc
        self.path_regex = re.compile(path_regex) if path_regex else None
        self.workers = workers
        self.expire_after = expire_after
        self.node_size = node_size
        self.layout = layout or {"physics": True, "hierarchical": False}
        self.show_menu = show_menu
        self.graph = nx.DiGraph()
        
        try:
            self.results = self._prev_results()
        except:
            pass
        
    # Main methods
    def detect_communities(self):
        from networkx.algorithms import community
        return list(community.greedy_modularity_communities(self.graph))

    def fetch_page(self, url: str, depth: int, session: CachedSession):
        """Fetch a page, extract valid links, and return node data with found links."""
        logger.info(f"Fetching (depth {depth}): {url}")
        try:
            response = session.get(url, timeout=5)
            response.raise_for_status()
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            links = set()
            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link["href"])
                normalized_url = self._normalize_url(full_url)
                if self._is_valid_url(normalized_url):
                    links.add(normalized_url)
                    if len(links) >= self.max_links:
                        break
            node_data = {
                "title": self._get_article_title(url),
                "label": self._get_article_title(url),
                "status_code": response.status_code,
                "depth": depth
            }
            return node_data, links
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            logger.error(f"HTTP error {status_code} for {url}: {str(e)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error for {url}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {str(e)}")
        # In case of error, return minimal data with no links.
        node_data = {
            "title": self._get_article_title(url),
            "label": self._get_article_title(url),
            "status_code": None,
            "depth": depth
        }
        return node_data, set()

    def process_url(self, url: str, depth: int, session: CachedSession, visited: set):
        """Process a URL and update the graph. Always return a set of links for further processing."""
        if url in visited:
            return set()
        visited.add(url)
        node_data, links = self.fetch_page(url, depth, session)
        # Remove self-loops and avoid duplicate edges
        valid_links = {link for link in links if link != url}
        self.graph.add_node(url, **node_data)
        for link in valid_links:
            # Avoid duplicate edge creation
            if not self.graph.has_edge(url, link):
                self.graph.add_edge(url, link)
        # Only return links for further processing if within max_depth.
        return valid_links if depth < self.max_depth else set()

    def crawl(self):
        """Crawl the website using a ThreadPoolExecutor."""
        try:
            print(f'Predicted time for crawling: {self._predict_time(self.max_depth,self.max_links,self.workers)}')
        except:
            pass # Because it only works when _prev_results works
        
        
        self.start_crawl_time = time.time()
        logger.info("Starting crawl")
        session = CachedSession(
            cache_name=f"cache/{self.domain}",
            expire_after=self.expire_after,
            allowable_methods=("GET",)
        )
        session.verify = True
        
        visited = set()
        frontier = [(self.start_url, 0)]
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while frontier:
                futures = {}
                for url, depth in frontier:
                    if url not in visited:
                        future = executor.submit(self.process_url, url, depth, session, visited)
                        futures[future] = depth
                frontier = []
                for future in futures:
                    links = future.result() or set()
                    current_depth = futures[future]
                    if current_depth < self.max_depth:
                        for link in links:
                            if link not in visited:
                                frontier.append((link, current_depth + 1))
        self.end_crawl_time = time.time()                    
        # Нужно закрыть сессию)
        session.close()
        logger.info("Crawl complete")
         
    def visualize(self, rebuild: bool = False, force_reload: bool = False, force_file_name : str = ''):
        if rebuild or force_reload or not self.graph.nodes:
            self.crawl()
        degrees = [self.graph.in_degree(n) for n in self.graph.nodes()]
        self.min_degree = min(degrees) if degrees else 0
        self.max_degree = max(degrees) if degrees else 1
        logger.info("Starting visualization")

        net = Network(
            notebook=False, 
            directed=True, 
            height="800px", 
            width="100%", 
            cdn_resources="remote",
            bgcolor = '#000000',
            font_color = '#ffffff',
            filter_menu = self.show_menu
        )

        
        # Example custom physics to help prevent node overlap
        net.repulsion(
            node_distance=500,
            central_gravity=0.2,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09
        )
        net.font_color
        #net.set_options(json.dumps(options))
        for node in self.graph.nodes:
            data = self.graph.nodes[node]
            status_code = data.get("status_code", 0)
            in_degree = self.graph.in_degree(node)
            tooltip = (
                f"<div style='padding: 8px; background: #e6ffe6'>"
                f"<b>URL:</b> {node}<br>"
                f"<b>Status:</b> {status_code}<br>"
                f"<b>In-Degree:</b> {in_degree}<br>"
                f"</div>"
            )
            title = data.get("label", node)
            color = "#ffcccc" if status_code and 400 <= int(status_code) < 600 else self._get_node_color(node)
            full_url = node if node.startswith(("http://", "https://")) else f"http://{node}"
            escaped_url = full_url.replace("'", "\\'")
            net.add_node(
                node,
                label=self._get_article_title(title),
                title=tooltip,
                size=self._calculate_node_size(node),
                font={"size": self._calculate_node_size(node)},
                color=color,
                url=full_url,
                onclick=f"window.open('{escaped_url}', '_blank');",
                shapeProperties={"allowHtml": True}
            )

        # Avoid duplicate edges by checking existing net.edges (list of dicts)
        for edge in self.graph.edges:
            src, dst = edge
            if src != dst:
                # Check if there's an existing edge from src to dst
                if not any(e["from"] == src and e["to"] == dst for e in net.edges):
                    net.add_edge(
                        src,
                        dst,
                        color={
                            "color": "#2B7CE9",
                            "highlight": "#FF0000",
                            "hover": "#FF0000"
                            },
                        selectionWidth=3,
                        smooth=False
                        )
                    
        ipynb_dir =  '\\'.join(get_this_ipynb().split('\\')[:-1])
        directory = ensure_directory_exists(ipynb_dir + '\\graphs')
        
        if not len(force_file_name):
            
            try:
                graph_num = max([int(i.split('.')[0].replace('graph','')) for i in list_files(directory) if 'graph' in i.split('.')[0]])+1
            except:
                graph_num = 0
            
            file = f"{directory}\\graph{graph_num}.html"
            self.save_graph(filename=f'{directory}\\graph{graph_num}.pkl')
        else:
            file = f"{directory}\\{force_file_name}.html"
            self.save_graph(filename=f"{directory}\\{force_file_name}.pkl")
        logger.info("Сохранение графа в HTML-файл")
        
        
        try:
            text = f'{file} | max_depth: {self.max_depth} | max_links: {self.max_links} | crawl time: {self.end_crawl_time - self.start_crawl_time} | workers: {self.workers}'
        except:
            text = f'{file} | max_depth: {self.max_depth} | max_links: {self.max_links} | workers: {self.workers}'
        net.write_html(file, open_browser=True)
        append_to_file(ipynb_dir+f'\\{self.__name__().split('.')[-1]}.txt',text)
        
        
        logger.info(f"Graph saved as {file} and opened in browser")
        
        
        print(text)
    
    def save_graph(self, filename="website_graph.pkl"):
            """Save the current graph state and parameters to a pickle file."""
            data = {
                "graph": self.graph,
                "start_url": self.start_url,
                "max_depth": self.max_depth,
                "max_links": self.max_links,
                "domain": self.domain,
                "path_regex": self.path_regex.pattern if self.path_regex else None,
                "workers": self.workers,
                "expire_after": self.expire_after,
                "node_size": self.node_size,
                "layout": self.layout,
            }
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            print(f"Graph saved to {filename}")

    def load_graph(self, filename="website_graph.pkl"):
        """Load the graph state and parameters from a pickle file and update self."""
        try:
            ipynb_dir =  '\\'.join(get_this_ipynb().split('\\')[:-1])
            directory = ensure_directory_exists(ipynb_dir + '\\graphs')
            with open(f'{directory}\\{filename}', "rb") as f:
                data = pickle.load(f)
        except:
            with open(f'{filename}', "rb") as f:
                data = pickle.load(f)
                
        self.graph = data.get("graph", self.graph)
        self.start_url = data.get("start_url", self.start_url)
        self.max_depth = data.get("max_depth", self.max_depth)
        self.max_links = data.get("max_links", self.max_links)
        self.domain = data.get("domain", self.domain)
        regex_pattern = data.get("path_regex")
        self.path_regex = re.compile(regex_pattern) if regex_pattern else None
        self.workers = data.get("workers", self.workers)
        self.expire_after = data.get("expire_after", self.expire_after)
        self.node_size = data.get("node_size", self.node_size)
        self.layout = data.get("layout", self.layout)
        print(f"Graph loaded from {filename}")

    # Inner methods
    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/'),
            "",
            parsed.query,
            ""
        ))
        return normalized

    def _is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path
        # Define disallowed prefixes for Wikipedia pages
        disallowed_prefixes = ["/wiki/Special", "/wiki/Help", "/wiki/Portal", "/wiki/Wikipedia"]
        # Check if the URL path starts with any of these prefixes
        if any(path.startswith(prefix) for prefix in disallowed_prefixes):
            return False
        # Also avoid main page and ensure it's within the desired domain
        is_main_page = path.lower().endswith("main_page")
        valid = (self.domain in parsed.netloc and not is_main_page and 
                (not self.path_regex or self.path_regex.search(path)))
        return valid

    def _get_article_title(self, url: str) -> str:
        match = re.search(r'/wiki/([^/]+)', url)
        if match:
            return match.group(1).replace('_', ' ')
        return url.split('//')[-1].split('/')[0]

    def _calculate_node_size(self, node: str) -> int:
        in_degree = self.graph.in_degree(node)
        size = 10 + 3 * in_degree if self.node_size == "degree" else 10
        return size

    def _get_node_color(self, node: str) -> str:
        
        if self.max_degree == self.min_degree:
            return "#D4EDD4"
        t = (self.graph.in_degree(node) - self.min_degree) / (self.max_degree - self.min_degree)
        h = 0.33 - 0.25 * t
        s = 0.8 + 0.2 * t
        v = 0.9 + 0.1 * t
        r, g, b = hsv_to_rgb(h, s, v)
        return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    
    def _prev_results(self):
        try:
            with open('WebsiteGraphMP.txt') as f:
                lines = f.readlines()
        except:
            lines = read_github_txt_file(r'https://raw.githubusercontent.com/Ackrome/WebsiteGraph/refs/heads/main/WebsiteGraphMP.txt',True,'WebsiteGraphMP.txt').split('\n')

        names = ['name','max_depth', 'max_links','crawl_time','workers']


        results = []
        for i in list(map(lambda x: x.split("|"), lines)):
            if len(i) == len(names):
                results.append([_.split(':')[-1] for _ in i])
        results = pd.DataFrame(results, columns=names)
        results['name'] = results['name'].transform(lambda x: x.split('\\')[-1])
        results['max_depth'] = pd.to_numeric(results['max_depth'])
        results['max_links'] = pd.to_numeric(results['max_links'])
        results['crawl_time'] = pd.to_numeric(results['crawl_time'])
        results['workers'] = pd.to_numeric(results['workers'].transform(lambda x: x[:-1]))
        results.set_index('name', inplace=True)
        results.sort_values('crawl_time', inplace=True)

        return results

    def _predict_time(self,max_depth,max_links,workers):
        gran_mean_times = self.results.groupby(['max_depth','max_links','workers'])[['crawl_time']].agg(['mean'])
        noworkers_mean_times = self.results.groupby(['max_depth','max_links'])[['crawl_time']].agg(['mean'])
        
        if (max_depth,max_links,workers) in gran_mean_times.index:
            return gran_mean_times.loc[max_depth,max_links,workers].iloc[0]
        
        elif (max_depth,max_links) in noworkers_mean_times.index:
            return noworkers_mean_times.loc[max_depth,max_links].iloc[0]
        
        else:
            X = self.results[['max_depth','max_links','workers']].values
            y = self.results['crawl_time'].values

            X = PolynomialFeatures(degree=2).fit_transform(X)
            model1 = OLS(y,X)
            model1 = model1.fit()


            mean_times = self.results[self.results.max_links==10].groupby('max_depth')[['crawl_time']].agg(['mean'])
            x = PolynomialFeatures(degree=2).fit_transform(mean_times.index.values.reshape(-1,1))

            model2 = OLS(mean_times.values,x)
            model2 = model2.fit()

            return (model1.predict(PolynomialFeatures(degree=2).fit_transform([[max_depth,max_links,workers]]))[0] + model2.predict(PolynomialFeatures(degree=2).fit_transform([[max_depth]]))[0])/2
        
    # Magic Methods
    def __str__(self):
        return (f"WebsiteGraphMP(start_url='{self.start_url}', nodes={self.graph.number_of_nodes()}, "
                f"edges={self.graph.number_of_edges()})")

    def __repr__(self):
        return (f"WebsiteGraphMP(start_url='{self.start_url}', max_depth={self.max_depth}, "
                f"max_links={self.max_links}, workers={self.workers})")

    def __eq__(self, other):
        if not isinstance(other, WebsiteGraphMP):
            return NotImplemented
        # Compare start_url and basic graph structure (nodes and edges)
        return (self.start_url == other.start_url and
                nx.is_isomorphic(self.graph, other.graph))

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __add__(self, other):
            if not isinstance(other, WebsiteGraphMP):
                return NotImplemented

            # Use NetworkX's compose to get the union of the two graphs.
            # (Nodes present in both graphs will be merged; edge sets are united.)
            new_graph = nx.compose(self.graph, other.graph)

            # Invent union logic for parameters:
            # For example, use the start_url from self, and pick the more permissive (max) values.
            new_max_depth = max(self.max_depth, other.max_depth)
            new_max_links = max(self.max_links, other.max_links)
            new_workers = max(self.workers, other.workers)
            new_expire_after = max(self.expire_after, other.expire_after)

            # For regex pattern, choose self's pattern if available, otherwise other.
            new_path_regex = None
            if self.path_regex and other.path_regex:
                new_path_regex = self.path_regex.pattern if len(self.path_regex.pattern) >= len(other.path_regex.pattern) else other.path_regex.pattern
            elif self.path_regex:
                new_path_regex = self.path_regex.pattern
            elif other.path_regex:
                new_path_regex = other.path_regex.pattern

            # Create a new WebsiteGraph instance with unioned parameters.
            new_instance = WebsiteGraphMP(
                start_url=self.start_url,  # you can decide which one to use
                max_depth=new_max_depth,
                max_links=new_max_links,
                path_regex=new_path_regex,
                workers=new_workers,
                expire_after=new_expire_after,
                layout=self.layout  # or merge layouts if needed
            )
            # Set the union graph
            new_instance.graph = new_graph

            return new_instance
    
    def __iadd__(self, other):
        # In-place union: merge other's graph into self
        if not isinstance(other, WebsiteGraphMP):
            return NotImplemented
        self.graph = nx.compose(self.graph, other.graph)
        self.max_depth = max(self.max_depth, other.max_depth)
        self.max_links = max(self.max_links, other.max_links)
        self.workers = max(self.workers, other.workers)
        self.expire_after = max(self.expire_after, other.expire_after)
        # For path_regex and layout, you can choose to keep self's parameters.
        return self

    def __sub__(self, other):
        if not isinstance(other, WebsiteGraphMP):
            return NotImplemented
        # Subtract nodes found in the other graph from self.graph
        new_instance = WebsiteGraphMP(
            start_url=self.start_url,
            max_depth=self.max_depth,
            max_links=self.max_links,
            path_regex=self.path_regex.pattern if self.path_regex else None,
            workers=self.workers,
            expire_after=self.expire_after,
            layout=self.layout
        )
        new_instance.graph = self.graph.copy()
        for node in other.graph.nodes():
            if node in new_instance.graph:
                new_instance.graph.remove_node(node)
        return new_instance

    def __iter__(self):
        # Iterate over nodes as (node, attributes) tuples.
        return iter(self.graph.nodes(data=True))

    def __len__(self):
        return self.graph.number_of_nodes()

    def __getitem__(self, key):
        # Allow indexing by node ID to get node attributes.
        return self.graph.nodes[key]

    def __contains__(self, key):
        return key in self.graph

    def __bool__(self):
        return self.graph.number_of_nodes() > 0

    def __call__(self):
        # Calling the instance triggers a re-crawl.
        self.crawl()
        return self        
    
    def __name__(self):
        return 'WebsiteGraphMP'