import datetime
import argparse
import cv2
import ultralytics
import json
import os


__doc__ = '''Основной файл программного модуля анализа данных для обеспечения безопасности в зоне взлета / посадки на основе технического зрения и искусственного интеллекта'''



def parse_arguments():
    '''Функция для парсинга аргументов, полученных при вызове через $ python main.py --example example ...'''

    parser = argparse.ArgumentParser(description='landing-takeoff-safety')
    parser.add_argument('--data_type', type=bool, action='store_true', help='Тип входных файлов: Файлы, если --data_type указан как параметр при запуске, иначе - Видеопоток с девайса.')
    parser.add_argument('--data', type=str, required=True, help='Расположение данных.')
    parser.add_argument('--weights_path', type=str, required=True, help='Расположение файла с весами модели.')
    parser.add_argument('--roi', type=str, required=True, help='Region-of-Interest "x,y,w,h".')
    parser.add_argument('--output_path', type=str, required=True, help='Путь к результирующему JSON файлу.')
    parser.add_argument('--visualization', type=bool, action='store_true', help='Если при запуске --visualization указан как параметр, то открывается дополнительное окно OpenCV с визуализацией распознавания.')
    return parser.parse_args()



def load_yolo_model(weights_path):
    '''Функция загрузки метода и весов для дальнейшей детекции из библиотеки ultralytics'''
    return ultralytics.YOLO(model=weights_path)



def detect_objects_in_files(model, data, roi, output_path, visualization):
    '''Функция детекции объектов на изображениях и видео'''

    # Преобразование ROI в кортеж
    roi = tuple(map(int, roi.split(',')))

    # Проверка, является ли источник директорией. Если да, то для анализа будут использованы все поддерживаемые изображения и видео
    if os.path.isdir(data):
        files = [os.path.join(data, f) for f in os.listdir(data) if f.endswith(('.jpg', '.jpeg', '.png', 'bmp', 'tiff', '.mp4', '.avi'))]
    else:
        files = [data] if data.endswith(('.jpg', '.jpeg', '.png', 'bmp', 'tiff', '.mp4', '.avi')) else []

    results_list = []
    frames = []
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            frame = cv2.imread(file)
            if frame is None:
                raise ValueError(f"Unable to load image {file}")
            frames.append(frame)
        else:
            cap = cv2.VideoCapture(file)
            if not cap.isOpened():
                raise ValueError(f"Unable to open video source {file}")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
    
    # Если подключена визуализация, то открывается окно opencv
    if visualization:
        for frame_idx, frame in enumerate(frames):
            # Обрезание ROI
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w]

            # Детекция объектов
            results = model(roi_frame)

            # Проверка наличия распознанных объектов в зоне интереса
            if len(results) > 0 and len(results[0].boxes.xyxy) > 0:
                for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Создание элемента списка JSON файла с результатами
                json_data = {
                    'file': file,
                    'frame_idx': frame_idx,
                    'detections': results[0].boxes.xyxy.cpu().numpy().tolist(),
                    'scenario': "Замечено препятствие в проверяемой области, требуется присутствие оператора!"
                }
                results_list.append(json_data)

                print("Замечено препятствие в проверяемой области, требуется присутствие оператора!")

            cv2.imshow('Landing / Takeoff safety', cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3))
            cv2.waitKey(1)
        cv2.destroyAllWindows()
    else:
        for frame_idx, frame in enumerate(frames):
            # Обрезание ROI
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w]

            # Детекция объектов
            results = model(roi_frame)
            
            # Проверка наличия распознанных объектов в зоне интереса
            if len(results[0].boxes['xyxy']) > 0:
                # Создание JSON файла с результатами
                json_data = {
                        'file': file,
                        'frame_idx': frame_idx,
                        'detections': results[0].boxes.xyxy.cpu().numpy().tolist(),
                        'scenario': "Замечено препятствие в проверяемой области, требуется присутствие оператора!"
                }
                results_list.append(json_data)

                print("Замечено препятствие в проверяемой области, требуется присутствие оператора!")

    # Сохранение всех результатов в один JSON файл
    with open(output_path, 'w') as json_file:
        json.dump(results_list, json_file, indent=4)

    

def detect_from_device(model, data, roi, output_path):
    '''Функция детекции объектов на девайсе (камере)'''

    # Преобразование ROI в кортеж
    roi = tuple(map(int, roi.split(',')))

    # Открытие девайса (камеры)
    cap = cv2.VideoCapture(int(data))
    if not cap.isOpened():
        raise ValueError("Unable to open device")
    
    results_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обрезание ROI
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]

        # Детекция объектов
        results = model(roi_frame)

        # Проверка наличия объектов
        if len(results) > 0 and len(results[0].boxes.xyxy) > 0:
            for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(roi_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Создание результирующего элемента в JSON файле с привязкой по времени
            json_data = {
                'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z'),
                'detections': results[0].boxes.xyxy.cpu().numpy().tolist(),
                'scenario': "Замечено препятствие в проверяемой области, требуется присутствие оператора!"
            }
            results_list.append(json_data)

            print("Замечено препятствие в проверяемой области, требуется присутствие оператора!")

        cv2.imshow('Landing / Takeoff safety', cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2))
        
        if cv2.waitKey(1)&0xFF==ord('q'):
            break    

    cap.release()
    cv2.destroyAllWindows()

    # Сохранение всех результатов в один JSON файл
    with open(output_path, 'w') as json_file:
        json.dump(results_list, json_file, indent=4)



def main():
    '''Основная функция'''

    args = parse_arguments()
    model = load_yolo_model(args.weights_path)

    if args.data_type:
        detect_objects_in_files(model, args.data, args.roi, args.output_path, args.visualization)
    else:
        detect_from_device(model, args.roi, args.output_path)


if __name__ == '__main__':
    main()