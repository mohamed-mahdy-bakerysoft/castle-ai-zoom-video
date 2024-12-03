# -*- coding: utf-8 -*-
# @Time    : 24.03.2022 09:52
# @Author  : Gevorg Minasyan
# @Email   : gevorg@podcastle.ai
# @File    : worker.py

import os
import time
import threading
from abc import ABC, abstractmethod
from google.cloud import pubsub_v1

from span_api_splitter import logger, literals


class Worker(ABC):

    @abstractmethod
    def run(self):
        pass


class PubSubWorker(Worker, threading.Thread):

    def __init__(
        self,
        topic_id,
        subscriber: pubsub_v1.SubscriberClient,
        message_processor: callable,
    ):
        threading.Thread.__init__(self)
        self.subscriber = subscriber
        self.max_messages = int(os.getenv(literals.MAX_MESSAGES))
        self.message_processor = message_processor
        self.subscriber = subscriber
        self.subscription_path = subscriber.subscription_path(
            os.getenv(literals.PROJECT_ID), topic_id
        )
        self.running = True
        self.unhealthy_file_path = literals.UNHEALTHY_FILE_PATH
        flow_control = pubsub_v1.types.FlowControl(max_messages=self.max_messages)
        self.streaming_pull_future = subscriber.subscribe(
            self.subscription_path, callback=self.callback, flow_control=flow_control
        )

    def callback(self, message: pubsub_v1.subscriber.message.Message) -> None:
        message.ack()
        self.message_processor(message.data)

    def run(self):
        with self.subscriber:
            try:
                # When `timeout` is not set, result() will block indefinitely,
                # unless an exception is encountered first.
                self.streaming_pull_future.result()
            except Exception:
                logger.exception(f"error occurred in pubsub worker: {str(e)}")
                self.streaming_pull_future.cancel()  # Trigger the shutdown.
                self.streaming_pull_future.result()  # Block until the shutdown is complete.
                with open(self.unhealthy_file_path, "w") as file:
                    pass

    def stop(self):
        self.streaming_pull_future.cancel()
        self.streaming_pull_future.result(timeout=30)
        self.running = False


class MainPubSubWorker(Worker):

    def __init__(
        self,
        message_processor: callable,
    ):

        subscriber = pubsub_v1.SubscriberClient()
        self.workers = [
            PubSubWorker(
                subscriber=subscriber,
                topic_id=os.getenv(literals.SUBSCRIPTION_ID),
                message_processor=message_processor,
            )
            for _ in range(int(os.getenv(literals.NUM_WORKERS)))
        ]

    def run(self):
        logger.info("worker stared")
        for it in self.workers:
            it.start()

        while True:
            statuses = [it.is_alive() for it in self.workers]
            if sum(statuses) == 0:
                break
            time.sleep(1)

    def stop(self):
        logger.info("slave workers stopping...")
        for it in self.workers:
            it.stop()
