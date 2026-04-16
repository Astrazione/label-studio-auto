import os
import uuid
from io import BytesIO
from urllib.parse import urljoin

import requests
import yaml
from PIL import Image
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase


DEFAULT_CONFIG_PATH = "/app/config.yaml"


class YOLOv12Backend(LabelStudioMLBase):
    def _load_config(self):
        if hasattr(self, "config"):
            return self.config

        config_path = os.getenv("CONFIG_PATH", DEFAULT_CONFIG_PATH)

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        return self.config

    def _get_model(self):
        if hasattr(self, "model"):
            return self.model

        config = self._load_config()

        model_path = os.getenv(
            "MODEL_PATH",
            config["model"].get("path", "/app/weights/best.pt"),
        )

        self.model = YOLO(model_path)
        return self.model

    def _get_confidence_threshold(self):
        config = self._load_config()

        return float(
            os.getenv(
                "CONFIDENCE_THRESHOLD",
                config["model"].get("confidence_threshold", 0.25),
            )
        )

    def _get_model_version(self):
        config = self._load_config()
        return config["model"].get("version", "unknown")

    def _get_labels(self):
        config = self._load_config()

        # YAML может прочитать ключи как int, но на всякий случай приводим явно
        return {int(class_id): label for class_id, label in config["labels"].items()}

    def _get_label_studio_url(self):
        config = self._load_config()

        return os.getenv(
            "LABEL_STUDIO_URL",
            config["label_studio"].get("url", "http://label-studio:8080")
        )

    def _resolve_image_url(self, image_url):
        if image_url.startswith(("http://", "https://")):
            return image_url

        return urljoin(self._get_label_studio_url(), image_url)

    def _get_auth_headers(self):
        api_key = os.getenv("LABEL_STUDIO_API_KEY")

        if not api_key:
            return {}

        return { "Authorization": f"Token {api_key}" }

    def _load_image(self, url):
        response = requests.get(url, headers=self._get_auth_headers())
        response.raise_for_status()

        return Image.open(BytesIO(response.content)).convert("RGB")

    @staticmethod
    def _xyxy_to_label_studio_bbox(x1, y1, x2, y2, image_width, image_height):
        return {
            "x": x1 / image_width * 100,
            "y": y1 / image_height * 100,
            "width": (x2 - x1) / image_width * 100,
            "height": (y2 - y1) / image_height * 100,
            "rotation": 0,
        }

    def _make_region(self, box, label, image_width, image_height):
        config = self._load_config()
        prediction_config = config["prediction"]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])

        bbox = self._xyxy_to_label_studio_bbox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            image_width=image_width,
            image_height=image_height,
        )

        return {
            "id": str(uuid.uuid4()),
            "from_name": prediction_config.get("from_name", "label"),
            "to_name": prediction_config.get("to_name", "image"),
            "type": prediction_config.get("type", "rectanglelabels"),
            "value": {
                **bbox,
                "rectanglelabels": [label],
            },
            "score": score,
        }

    def _predict_single_task(self, task):
        model = self._get_model()
        labels = self._get_labels()
        confidence_threshold = self._get_confidence_threshold()

        image_url = self._resolve_image_url(task["data"]["image"])
        image = self._load_image(image_url)

        image_width, image_height = image.size

        result = model.predict(
            image,
            conf=confidence_threshold,
            verbose=False,
        )[0]

        regions = []
        scores = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id not in labels:
                    continue

                region = self._make_region(
                    box=box,
                    label=labels[class_id],
                    image_width=image_width,
                    image_height=image_height,
                )

                regions.append(region)
                scores.append(region["score"])

        return {
            "result": regions,
            "score": sum(scores) / len(scores) if scores else 0,
            "model_version": self._get_model_version(),
        }

    def predict(self, tasks, context=None, **kwargs):
        return [self._predict_single_task(task) for task in tasks]