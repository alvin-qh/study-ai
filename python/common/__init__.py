from .config import Config
from .log import log

conf = Config()
conf.from_pyfile('conf/conf.py')
