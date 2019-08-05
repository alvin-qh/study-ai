import logging
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        },
        'short': {
            'format': '%(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    }
})

log = logging.getLogger()
