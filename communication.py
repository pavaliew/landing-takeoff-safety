# -*- coding: utf-8 -*-
import pymavlink.mavutil as utility
import pymavlink.dialects.v20.all as dialect


class MAVLinkCommunication:
    '''Класс для работы с MAVLink коммуникацией'''
    
    def __init__(self, port='COM12', source_system=1, source_component=0):
        '''
        Инициализация MAVLink соединения
        
        :param port: Порт подключения (например, 'COM12' или '/dev/ttyUSB0')
        :param source_system: ID системы-источника
        :param source_component: ID компонента-источника
        '''
        self.port = port
        self.connection = utility.mavlink_connection(
            device=port,
            source_system=source_system,
            source_component=source_component
        )
        self.connection.wait_heartbeat()
        print(f"MAVLink подключен к {port}")
    
    def send_detection_alert(self, detection_count, class_names, severity=dialect.MAV_SEVERITY_CRITICAL):
        '''
        Отправка уведомления о детекции на GCS
        
        :param detection_count: Количество обнаруженных объектов
        :param class_names: Список классов обнаруженных объектов
        :param severity: Уровень критичности сообщения
        '''

        if detection_count > 0:
            message_text = f"OBSTACLE DETECTED: {detection_count} object(s) - {', '.join(class_names)}"
        else:
            message_text = "Area clear"
        
        # Ограничение длины сообщения, т.к. в MAVLink STATUSTEXT максимум 50 символов
        message_text = message_text[:50]
        
        message = dialect.MAVLink_statustext_message(
            severity=severity,
            text=message_text.encode("utf-8")
        )
        
        self.connection.mav.send(message)
        print(f"MAVLink message sent: {message_text}")
    
    def change_to_loiter(self):
        '''
        Переключение дрона в режим LOITER при обнаружении препятствия
        
        :return: True если успешно, False если ошибка
        '''
        
        try:
            flight_modes = self.connection.mode_mapping()
            loiter_id = flight_modes.get("LOITER")
            
            if loiter_id is None:
                print("LOITER mode not available")
                return False
            
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                176,  # MAV_CMD_DO_SET_MODE
                0,
                1,
                loiter_id,
                0, 0, 0, 0, 0
            )
            
            print("Команда LOITER отправлена")
            return True
        
        except Exception as e:
            print(f"Ошибка при переключении в LOITER: {e}")
            return False
    
    def close(self):
        '''Закрытие соединения'''

        if self.connection:
            self.connection.close()
            print("MAVLink соединение закрыто")
