# label-studio-auto
Label studio auto annotation based on yolo detection model

Реализован ML backend для интеграции модели детекции YOLOv12s с Label Studio.
Модель автоматически предсказывает bounding boxes для классов:
- `chip`
- `gap`
- `protrusion`

Пользователь может использовать предсказания как pre-annotations и доразмечать изображения вручную.

## Структура проекта

```
.
├── docker-compose.yml
├── label-studio-data/
└── ml-backend/
    ├── Dockerfile
    ├── requirements.txt
    ├── model.py
    ├── config.yaml
    └── weights/
        └── best.pt
```

## Конфигурация модели
Основные параметры задаются в `ml-backend/config.yaml`:
```yaml
model:
  path: /app/weights/best.pt
  confidence_threshold: 0.25
  version: v12s-custom-v0.1

label_studio:
  url: http://label-studio:8080

prediction:
  from_name: label
  to_name: image
  type: rectanglelabels

labels:
  0: chip
  1: gap
  2: protrusion
```

## Запуск
Положите веса модели в: 
```
ml-backend/weights/best.pt`
```
Запустите сервисы
```bash
docker compose up --build
```
Label Studio будет доступен по адресу: `http://localhost:8080`

ML backend (не имеет GUI): `http://localhost:9090`
### Настройка Label Studio
В Label Studio создайте проект и укажите labeling config:
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="chip"/>
    <Label value="gap"/>
    <Label value="protrusion"/>
  </RectangleLabels>
</View>
```
#### Важно: значения from_name и to_name (а также названия классов) в config.yaml должны совпадать с name в XML-конфиге Label Studio.
### Подключение ML Backend
- В Label Studio перейдите `Organization → API Tokens Settings` и включите `Legacy Tokens`, сохраните изменения
- Перейдите во вкладку `Account & Settings → Legacy Token`, скопируйте Access Token и вставьте в поле `LABEL_STUDIO_API_KEY` в `docker-compose.yaml`
- Пересоберите проект командой `docker compose up --build`
- В проекте Label Studio откройте: `Settings → Model`, нажмите `Connect Model`, введите любое название модели
- Добавьте backend URL: `http://ml-backend:9090` и включите `Interactive preannotations`, нажмина `Validate and save`
- Если подключаете backend с хоста (не из контейнера), используйте: `http://localhost:9090`
- В опциях модели выберите `Send Test Request → Send Request`
- Если в ответе статус 200, то модель готова к работе
- Если вышла ошибка, смотрите логи контейнера ml-backend