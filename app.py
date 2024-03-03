import face_recognition
import os
import time
from datetime import datetime

KNOWN_FACES_DIR = 'known_faces'

# Cargar las imágenes de referencia y generar sus encodings
known_faces = []
known_names = []

start_time = time.time()  # Iniciar tiempo de ejecución

for filename in os.listdir(KNOWN_FACES_DIR):    
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image, model="large")[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(filename)[0])

print(f'Inicio de la ejecución: {datetime.now()}')

# Función para realizar el reconocimiento facial en una imagen de prueba
def recognize_faces(image_path):
    unknown_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        for match, name in zip(matches, known_names):
            if match:
                print(f"Se ha encontrado a {name} en la imagen.")
            else:
                print(f"No se ha encontrado a {name} en la imagen.")

# Ejemplo de uso
if __name__ == "__main__":
    image_path = 'test_image.jpeg'  # Ruta de la imagen de prueba
    recognize_faces(image_path)

end_time = time.time()  # Finalizar tiempo de ejecución
duration = end_time - start_time  # Duración de la ejecución
print(f'Fin de la ejecución: {datetime.now()}')
print(f'Duración de la ejecución: {duration} segundos')
