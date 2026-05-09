import cv2
import torch
from torchvision import transforms
from model import DeepfakeModel

model=DeepfakeModel()
model.load_state_dict(torch.load("deepfake_model.pth"))
model.eval()

transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while True:

    ret,frame=cap.read()

    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    face=transform(img).unsqueeze(0)

    with torch.no_grad():
        pred=model(face)
        prob=torch.sigmoid(pred).item()

    label="FAKE" if prob>0.5 else "REAL"

    color=(0,0,255) if label=="FAKE" else (0,255,0)

    cv2.putText(frame,label,(50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,color,2)

    cv2.imshow("Deepfake Detector",frame)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
