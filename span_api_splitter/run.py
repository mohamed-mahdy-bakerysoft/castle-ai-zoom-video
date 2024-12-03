import os
import time
import json
import signal
import traceback

from google.cloud import pubsub_v1

from span_api_splitter.utils import get_split_info
from span_api_splitter.subscriber import MainPubSubWorker
from span_api_splitter import (
    config,
    logger,
    literals
)


def publish_splitting_service_out_message(payload, project_id, topic_id, publisher):
    topic_path = publisher.topic_path(project_id, topic_id)
    data = json.dumps(payload).encode('utf-8')
    future = publisher.publish(topic_path, data)
    result = future.result()
    logger.debug(f'audio splitting out message published successfully, result: {result}')


class MessageProcessor:

    def __init__(self, client) -> None:
        self.client = client

    def process_message(self, data) -> None:
        try:
            logger.debug(f"received {data}.")
            start_time = time.time()
            if data and len(data) > literals.ZERO:
                payload = json.loads(data.decode('utf-8'))
                if not payload[literals.AUDIO_URL]:
                    raise ValueError('No audio received')
                result = get_split_info(payload, config)
                payload[literals.RESULT] = result
                publish_splitting_service_out_message(
                    payload=payload,
                    project_id=os.getenv(literals.PROJECT_ID),
                    topic_id=os.getenv(literals.TOPIC_OUT_ID),
                    publisher=self.client
                )
                end_time = time.time()
                process_time = round(end_time - start_time, literals.PRECISION)
                logger.debug(f"got result for {payload.get(literals.RESULT_ID)}, execution time: {process_time}s, points: "
                            f"{result.get('splittingPoints')}, silences: {result.get('silenceDurations')}.")
        except Exception as e:
            logger.error(f'error occurred at callback: {str(e)}')
            logger.error(traceback.format_exc())


subscriber = pubsub_v1.SubscriberClient()


def close_main_worker(
    signum,
    frame
):  
    logger.info('stoping main worker')
    worker.stop()


if __name__ == '__main__':
    publisher = pubsub_v1.PublisherClient()
    message_processor = MessageProcessor(publisher)
    worker = MainPubSubWorker(
        message_processor.process_message
    )
    signal.signal(
        signal.SIGTERM,
        close_main_worker
    )
    signal.signal(
        signal.SIGINT,
        close_main_worker
    )
    worker.run()