from rich.logging import RichHandler
from rich.progress import track as tqdm
import logging

console = RichHandler(show_path=False, log_time_format="[%X]")

def setup_logger(name='rich', level=logging.INFO, console=console):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console.setLevel(level)
    console.setFormatter(
        logging.Formatter(
            '%(message)s'
        )
    )
    logger.addHandler(console)
    logger.propagate = False
    return logger

logger = setup_logger()

def print(*args, **kwargs):
    if not any(args): 
        logger.info('\n')
        return
    logger.info(*args, **kwargs)
