
import base64
from io import BytesIO

from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn

import socketio
from aiohttp import web

# ============================================================
# 1. CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = 4567

MODEL_PATH = "pilotnet_steering_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"[DEVICE] Using {device}")


# ============================================================
# 2. PILOTNET ARCHITECTURE
# ============================================================

class PilotNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ELU(),

            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),

            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),

            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.ELU()
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),

            nn.Linear(1152, 116),
            nn.ELU(),

            nn.Linear(116, 50),
            nn.ELU(),

            nn.Linear(50, 10),
            nn.ELU(),

            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.fc_layers(
            self.conv_layers(x)
        )


# ============================================================
# 3. LOAD MODEL
# ============================================================

model = PilotNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("[MODEL] PilotNet loaded successfully")


# ============================================================
# 4. SOCKET.IO SERVER
# ============================================================

sio = socketio.AsyncServer(
    async_mode="aiohttp",
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True
)

app = web.Application()

sio.attach(app)


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_data):

    image = Image.open(
        BytesIO(base64.b64decode(image_data))
    ).convert("RGB")

    img_np = np.asarray(image)

    # Crop sky and hood
    if img_np.shape[0] <= 85:
        raise ValueError(
            f"Image height too small: {img_np.shape}"
        )

    img_np = img_np[60:-25, :, :]

    # Resize
    img_res = cv2.resize(
        img_np,
        (200, 66),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    img_res = img_res.astype(
        np.float32
    ) / 255.0

    # HWC -> CHW -> NCHW
    tensor = torch.from_numpy(
        np.ascontiguousarray(img_res)
    )

    tensor = tensor.permute(
        2, 0, 1
    ).unsqueeze(0)

    return tensor.to(device)


# ============================================================
# 6. MODEL INFERENCE
# ============================================================

def predict_steering(tensor):

    with torch.inference_mode():

        output = model(tensor)

    return float(output.item())


# ============================================================
# 7. SEND CONTROL
# ============================================================

async def send_control(
    sid,
    steering_angle,
    throttle
):

    control_data = {
        "steering_angle": str(steering_angle),
        "throttle": str(throttle)
    }

    print(
        f"[SEND] SID={sid} "
        f"Steering={steering_angle:.4f} "
        f"Throttle={throttle:.4f}"
    )

    await sio.emit(
        "steer",
        data=control_data,
        to=sid
    )


# ============================================================
# 8. SOCKET.IO EVENTS
# ============================================================

@sio.on("connect")
async def connect(sid, environ, auth=None):

    print(f"[CONNECT] Client connected: {sid}")

    print(
        f"[TRANSPORT] {sio.eio.transport(sid)}"
    )


@sio.on("disconnect")
async def disconnect(sid, reason=None):

    print(
        f"[DISCONNECT] Client disconnected: {sid}"
    )

    print(f"[REASON] {reason}")


@sio.on("telemetry")
async def telemetry(sid, data):

    if not data:

        print("[WARNING] Empty telemetry")

        await sio.emit(
            "manual",
            data={},
            to=sid
        )

        return

    try:

        # ----------------------------------------------------
        # A. Read telemetry
        # ----------------------------------------------------

        speed = float(data["speed"])

        image_data = data["image"]

        print(
            f"[TELEMETRY] Speed={speed:.2f}"
        )

        # ----------------------------------------------------
        # B. Preprocessing
        # ----------------------------------------------------

        tensor = preprocess_image(image_data)

        print(
            f"[IMAGE] Tensor shape: {tensor.shape}"
        )

        # ----------------------------------------------------
        # C. Inference
        # ----------------------------------------------------

        steering_angle = predict_steering(
            tensor
        )

        # ----------------------------------------------------
        # D. Vehicle control
        # ----------------------------------------------------

        throttle = 0.2 if speed < 12 else 0.0

        print(
            f"[PREDICTION] Steering={steering_angle:.4f}"
        )

        # ----------------------------------------------------
        # E. Send control
        # ----------------------------------------------------

        await send_control(
            sid,
            steering_angle,
            throttle
        )

    except Exception as e:

        print(
            f"[ERROR] Telemetry processing: {repr(e)}"
        )


# ============================================================
# 9. START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        f"[SERVER] Starting at http://127.0.0.1:{PORT}"
    )

    web.run_app(
        app,
        host=HOST,
        port=PORT
    )
