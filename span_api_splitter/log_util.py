import uuid
import logging


def generate_request_id(original_id: str = "") -> str:
    """
    Generate a new request ID, optionally including an original request ID
    """
    new_id = uuid.uuid4()

    if original_id:
        new_id = f"{original_id},{new_id}"

    return new_id


class RequestIdFilter(logging.Filter):
    """
    This is a logging filter that makes the request ID available for use in
    the logging format. Note that we're checking if we're in a request
    context, as we may want to log things before Flask is fully loaded.
    """

    def filter(self, record):
        record.request_id = generate_request_id()
        return True
