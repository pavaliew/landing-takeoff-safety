# landing-takeoff-safety

## Краткое описание

Программный модуль анализа данных для обеспечения безопасности в зоне взлёта / посадки на основе технического зрения и искусственного интеллекта.

Формат результирующего json-файла:

```json
[
    {
        "file": "path/to/file",
        "frame_idx": ...,
        "detections": [
            [
                ...,
                ...,
                ...,
                ...
            ]
        ],
        "scenario": "..."
    },
    ...
]
```

Данные:

* file – путь до файла, с которого взят кадр;
* frame_idx – id кадра в данном файле;
* detections: координаты прямоугольника в котором находится
  распознанный объект;
* scenario: сценарий для данной ситуации.

## Установка и запуск

1. Установите Python (рекомендуется версия 3.12.4).
2. Загрузите репозиторий программного модуля.
3. Установите требуемые библиотеки через команду, запущенную из директории с репозиторием: ``pip install -r requirements.txt``.
4. Запустите программный модуль через команду, указав необходимые аргументы. Пример:  

   ```
   python main.py --data_type --data test_data --roi "0,0,150,150" --output_path runs/run1.json
   ```

## Описание аргументов при запуске

* `--data_type` - тип входных данных: если указан при запуске как `python main.py --data_type`, то будут анализироваться файл с форматом PNG/BMP/TIFF/JPEG/MP4/AVI или директория с файлами определенных форматов по пути, прописанном в аргументе `--data`; если параметр не указан, то программа будет анализировать видеопоток в реальном времени с устройства (посадочной камеры), номер которого (начиная с 0) в аргументе `--data`.
* `--data` -  расположение директории с видеофайлами/изображениями или номер устройства (посадочной камеры), с которого будет анализироваться видеопоток. Пример:  

  ```
  python main.py --data 0
  ```
* `--weights_path` - расположение весов модели, поддерживаемой библиотекой Ultralytics. Пример:  

  ```
  python main.py --weights_path path/to/weights.pt
  ```
* `--roi` - границы области интереса кадра, задаваемые как левая верхняя точка прямоугольника и его высота и ширина. Значения указываются в формате (без пробелов): “x,y,w,h”. Пример:  

  ```
  python main.py --roi "0,0,200,200"
  ```
* `--output_path` - расположение JSON-файла, в который будет записано описание обнаруженных объектов в зоне
  взлета / посадки и сценарий безопасного полета. Пример:  

  ```
  python main.py --output_path run1.json
  ```
