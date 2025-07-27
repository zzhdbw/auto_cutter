from loguru import logger

logger.add("logs/vad.log", rotation="50 MB")
