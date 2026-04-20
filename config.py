import os
from dotenv import load_dotenv

load_dotenv()

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_CONSUMER_TOPIC    = os.getenv("KAFKA_CONSUMER_TOPIC", "pdf-topic")
KAFKA_PRODUCER_TOPIC    = os.getenv("KAFKA_PRODUCER_TOPIC", "pdf-response")
KAFKA_GROUP_ID          = os.getenv("KAFKA_GROUP_ID", "python-pdf-group")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set in .env")