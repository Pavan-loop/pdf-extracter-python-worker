import json
import logging
import time
import signal

from confluent_kafka import Consumer, Producer, KafkaException

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_TOPIC,
    KAFKA_PRODUCER_TOPIC,
    KAFKA_GROUP_ID,
)
from extractor import process_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_consumer() -> Consumer:
    conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": KAFKA_GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,          # FIX: manual commit only after handling
    }
    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_CONSUMER_TOPIC])
    logger.info("Kafka consumer connected successfully")
    return consumer


def create_producer() -> Producer:
    conf = {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS}
    logger.info("Kafka producer connected successfully")
    return Producer(conf)


def send_result(producer: Producer, job_id: str, result: dict):
    """Send successful extraction result back to Spring Boot via Kafka."""
    message = {
        "jobId": job_id,
        "status": "DONE",
        "documentType": result.get("documentType", "UNKNOWN"),  # FIX: use actual documentType from result
        "extractedData": result["extractedData"],
        "errorMessage": None,
        "processedAt": int(time.time() * 1000)
    }
    producer.produce(
        KAFKA_PRODUCER_TOPIC,
        key=job_id,
        value=json.dumps(message).encode("utf-8")
    )
    producer.poll(0)  # FIX: non-blocking, triggers delivery callbacks without stalling
    logger.info(f"Result sent to Kafka for jobId={job_id}")


def send_failure(producer: Producer, job_id: str, error: str):
    """Send failure message back to Spring Boot so user gets notified."""
    message = {
        "jobId": job_id,
        "status": "FAILED",
        "documentType": "UNKNOWN",
        "extractedData": None,
        "errorMessage": error,
        "processedAt": int(time.time() * 1000)
    }
    producer.produce(
        KAFKA_PRODUCER_TOPIC,
        key=job_id,
        value=json.dumps(message).encode("utf-8")
    )
    producer.poll(0)  # FIX: non-blocking, triggers delivery callbacks without stalling
    logger.warning(f"Failure sent to Kafka for jobId={job_id}: {error}")


def run():
    logger.info("=" * 50)
    logger.info("PDF Extraction Worker Starting")
    logger.info(f"Consuming from : {KAFKA_CONSUMER_TOPIC}")
    logger.info(f"Producing to   : {KAFKA_PRODUCER_TOPIC}")
    logger.info(f"Kafka broker   : {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info("=" * 50)

    consumer = create_consumer()
    producer = create_producer()

    running = True

    def handle_shutdown(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received, stopping worker...")
        running = False

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("Worker is ready. Waiting for PDFs...")

    try:                                           # FIX: guarantee cleanup on any exit path
        while running:
            try:
                # Poll for a message — 1 second timeout so we can check running flag
                msg = consumer.poll(timeout=1.0)

                if msg is None:
                    continue  # No message yet, keep waiting

                if msg.error():
                    logger.error(f"Kafka consumer error: {msg.error()}")
                    continue

                job_id = msg.key().decode("utf-8") if msg.key() else "unknown"
                message = json.loads(msg.value().decode("utf-8"))

                logger.info(f"Received message: jobId={job_id}")

                try:
                    pdf_path = message.get("pdfFilePath")
                    if not pdf_path:
                        raise ValueError("Message missing 'pdfFilePath' field")

                    logger.info(f"Processing file={pdf_path}, userId={message.get('userId')}")
                    result = process_pdf(pdf_path)
                    send_result(producer, job_id, result)

                except Exception as e:
                    # Single message failure — send FAILED, keep worker running
                    logger.error(f"Failed to process jobId={job_id}: {e}", exc_info=True)
                    send_failure(producer, job_id, str(e))

                finally:
                    # FIX: commit offset only after the message is fully handled (success or failure).
                    # With auto-commit disabled, a mid-processing crash will NOT advance the offset,
                    # so the message gets re-delivered on restart instead of being silently lost.
                    consumer.commit(message=msg, asynchronous=False)

            except KafkaException as e:
                logger.error(f"Kafka error: {e}", exc_info=True)
                time.sleep(2)

            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(2)

    finally:
        logger.info("Closing Kafka connections...")
        producer.flush()  # FIX: drain any buffered produce messages before exit
        consumer.close()
        logger.info("Worker stopped cleanly.")


if __name__ == "__main__":
    run()
