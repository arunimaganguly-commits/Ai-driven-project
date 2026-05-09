import cv2
import torch
from torchvision import transforms

from video_model import DeepfakeVideoModel
from face_detector import detect_face

# -----------------------------
# Load trained model
# -----------------------------

model = DeepfakeVideoModel()

model.load_state_dict(
    torch.load("video_model.pth", map_location="cpu")
)

model.eval()

# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# Start webcam
# -----------------------------

cap = cv2.VideoCapture(0)

frame_buffer = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Detect face
    # -----------------------------

    face = detect_face(rgb)

    if face is not None:

        # preprocess face
        face_tensor = transform(face)

        # add to sequence buffer
        frame_buffer.append(face_tensor)

        # keep only last 16 frames
        if len(frame_buffer) > 16:
            frame_buffer.pop(0)

        # -----------------------------
        # Run prediction
        # -----------------------------

        if len(frame_buffer) == 16:

            sequence = torch.stack(frame_buffer)

            # add batch dimension
            sequence = sequence.unsqueeze(0)

            with torch.no_grad():

                output = model(sequence)

                prob = torch.sigmoid(output).item()

            label = "FAKE" if prob > 0.5 else "REAL"

            color = (
                (0,0,255)
                if label == "FAKE"
                else (0,255,0)
            )

            cv2.putText(
                frame,
                f"{label} {prob:.2f}",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

    cv2.imshow("Video Deepfake Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()