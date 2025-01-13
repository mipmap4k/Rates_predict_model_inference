import yaml
import logging

with open('./config/logger.yaml', 'r', encoding='utf-8') as file:
    config_log = yaml.safe_load(file)

logging.basicConfig(
    filename=config_log['basic_logger']['filename'],
    filemode=config_log['basic_logger']['filemode'],
    format=config_log['basic_logger']['format'],
    datefmt=config_log['basic_logger']['datefmt'],
    level=logging.DEBUG,
    encoding=config_log['basic_logger'].get('encoding', 'utf-8') 
)

logger = logging.getLogger("basic_logger")
