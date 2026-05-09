import cv2

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

def detect_face(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50,50)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    face = image[y:y+h, x:x+w]

    return face