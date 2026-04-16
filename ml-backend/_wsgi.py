import os

from label_studio_ml.api import init_app
from model import YOLOv12Backend


app = init_app(
    model_class=YOLOv12Backend,
    redis_queue=os.getenv("RQ_QUEUE_NAME", "default"),
)