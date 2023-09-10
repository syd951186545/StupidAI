# 创建一个 logger
import logging
import config
from datetime import datetime


class log_print:
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    @staticmethod
    def info(x, *args):
        if not args:
            print(f"{log_print.current_time} - INFO - :{x}")
        else:
            print(f"{log_print.current_time} - INFO - :{x}", args)

    @staticmethod
    def debug(x, *args):
        if not args:
            print(f"{log_print.current_time} - DEBUG - :{x}")
        else:
            print(f"{log_print.current_time} - DEBUG - :{x}", args)

    @staticmethod
    def error(x, *args):
        if not args:
            print(f"{log_print.current_time} - ERROR - :{x}")
        else:
            print(f"{log_print.current_time} - ERROR - :{x}", args)


_logger = logging.getLogger('stupid_logger')
# 默认级别 DEBUG
_logger.setLevel(config.Global.LOGGER_LEVEL)
# 再创建一个处理器，用于输出到控制台
ch = logging.StreamHandler()
# 定义 handler 的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# 添加处理器到 logger
_logger.addHandler(ch)

logger = log_print() if config.Global.LOGGER == "print" else _logger
# 记录一条日志
logger.info(f'logger works with {config.Global.LOGGER}')