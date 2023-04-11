import logging
from omegaconf import DictConfig
import hydra

# A logger for this file
log = logging.getLogger(__name__)

# @hydra.main()
def bar():
    log.info("from bar")

@hydra.main()
def my_app(_cfg: DictConfig) -> None:
    log.info("Info level message")
    log.debug("Debug level message")
    bar()


if __name__ == "__main__":
    my_app()

