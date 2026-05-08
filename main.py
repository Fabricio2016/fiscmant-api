from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import gdown, io, os, base64, threading

app = FastAPI(title="FiscMant - API Unificada YOLO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("models", exist_ok=True)

# ── Configuracion de modelos ──────────────────────────────────────────────────
MODELS_CONFIG = {
    "safe_city":    {"drive_id": "1zWqG-Ydo8n37PWxePVESvuHxO5pUyxuk", "path": "models/safe_city.pt",    "conf": 0.10},
    "fo_nodo":      {"drive_id": "1tSSKvZesgWzhhX8vKBnk-7UoxkRTgKXI", "path": "models/fo_nodo.pt",      "conf": 0.10},
    "manguera":     {"drive_id": "1M6GnsAqOma8JzK5KkhhYugT_lXPsdc7M", "path": "models/manguera.pt",     "conf": 0.10},
    "cable":        {"drive_id": "12Gl0x6Y1DTlW7bge_nsJzTBBju-DF1un", "path": "models/cable.pt",        "conf": 0.10},
    "roseta":       {"drive_id": "1j6xOY3fb-1osgDkVia_yeqqlmXnxr11B", "path": "models/roseta.pt",       "conf": 0.10},
    "ventilador_1": {"drive_id": "1K5_IouYuaNGDKgv2Y7qSzzmCWBIIjcXk", "path": "models/ventilador_1.pt", "conf": 0.10},
    "ventilador_2": {"drive_id": "1ZUVqQ6_wqhuD02V-03XxaNDfXCEG9c1c", "path": "models/ventilador_2.pt", "conf": 0.10},
    "ventilador_3": {"drive_id": "1DdvtuRwS9XxveUTpNGzHTli-gwaBNLir", "path": "models/ventilador_3.pt", "conf": 0.10},
    "ventilador_4": {"drive_id": "12abF8szsggtzws4JBAqiB8ZnLWTTaaS6", "path": "models/ventilador_4.pt", "conf": 0.10},
    "ups":          {"drive_id": "12SHFp3842S5dfpuma7sGUzb9XFA4wVHI", "path": "models/ups.pt",          "conf": 0.10},
    "bateria":      {"drive_id": "1qVAzSbjMsYGj2mMRfi41jWB0hW1r8ItU", "path": "models/bateria.pt",      "conf": 0.10},
}

_models: dict = {}
_locks        = {key: threading.Lock() for key in MODELS_CONFIG}


def get_model(name: str) -> YOLO:
    """Carga el modelo solo la primera vez que se solicita (lazy loading)."""
    if name not in _models:
        with _locks[name]:
            if name not in _models:
                cfg  = MODELS_CONFIG[name]
                path = cfg["path"]
                if not os.path.exists(path):
                    print(f"[{name}] Descargando modelo...")
                    gdown.download(
                        f"https://drive.google.com/uc?id={cfg['drive_id']}",
                        path, quiet=False
                    )
                    print(f"[{name}] Descarga completa.")
                m = YOLO(path)
                m.overrides['imgsz']  = 640
                m.overrides['conf']   = cfg["conf"]
                m.overrides['device'] = 'cpu'
                _models[name] = m
                print(f"[{name}] Modelo listo. Clases: {m.names}")
    return _models[name]


def decode_image(image_base64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB").resize((640, 640))


def run_detection(model: YOLO, image: Image.Image, conf_threshold: float):
    results     = model(image, imgsz=640, conf=conf_threshold, verbose=False)
    detecciones = []
    clases      = []
    for r in results:
        for box in r.boxes:
            class_id  = int(box.cls)
            confianza = round(float(box.conf), 3)
            nombre    = model.names.get(class_id, f"clase_{class_id}")
            bbox      = [round(c, 1) for c in box.xyxy[0].tolist()]
            if confianza >= conf_threshold:
                detecciones.append({
                    "clase":         nombre,
                    "confianza":     confianza,
                    "confianza_pct": f"{round(confianza * 100)}%",
                    "bbox":          bbox
                })
                if nombre not in clases:
                    clases.append(nombre)
    return detecciones, clases


# ── Schemas ───────────────────────────────────────────────────────────────────

class DetectarRequest(BaseModel):
    image_base64: str
    confianza:    float = 0.25

class VentiladorRequest(BaseModel):
    image_base64:  str
    ventilador_id: int   = 1
    confianza:     float = 0.25


# ── Health ────────────────────────────────────────────────────────────────────

@app.api_route("/", methods=["GET", "HEAD"])
def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return JSONResponse({
        "status":   "ok",
        "servicio": "FiscMant API Unificada",
        "endpoints": [
            "POST /nodo-safe-city-cerrado/detectar",
            "POST /fo-nodo/detectar",
            "POST /manguera/detectar",
            "POST /cable/detectar",
            "POST /roseta/detectar",
            "POST /ventiladores/detectar",
            "POST /ups/detectar",
            "POST /bateria/detectar",
        ],
        "modelos_cargados": list(_models.keys())
    })


# ── Nodo Safe City Cerrado ────────────────────────────────────────────────────

@app.post("/nodo-safe-city-cerrado/detectar")
async def detectar_safe_city(req: DetectarRequest):
    model   = get_model("safe_city")
    image   = decode_image(req.image_base64)
    results = model(image, imgsz=640, verbose=False)

    CLASS_NAMES     = {0: "ETIQUETA", 1: "GABINET"}
    gabinete_valido = False
    etiqueta_valida = False
    conf_gabinete   = 0.0
    conf_etiqueta   = 0.0
    region_gabinete = None
    region_etiqueta = None
    coords_gabinete = None
    coords_etiqueta = None
    todas_det       = []

    for r in results:
        for box in r.boxes:
            class_id  = int(box.cls)
            confianza = round(float(box.conf), 3)
            bbox      = box.xyxy[0].tolist()
            nombre    = CLASS_NAMES.get(class_id, f"clase_{class_id}")
            x1, y1, x2, y2 = bbox
            cx     = (x1 + x2) / 2
            cy     = (y1 + y2) / 2
            region = f"{'superior' if cy < 320 else 'inferior'}_{'izquierda' if cx < 320 else 'derecha'}"

            todas_det.append({
                "clase":         nombre,
                "confianza":     confianza,
                "confianza_pct": f"{round(confianza * 100)}%",
                "region":        region,
                "bbox":          [round(c, 1) for c in bbox]
            })

            if class_id == 1 and confianza > conf_gabinete:
                gabinete_valido = confianza >= 0.40
                conf_gabinete   = confianza
                region_gabinete = region
                coords_gabinete = [round(c, 1) for c in bbox]

            if class_id == 0 and confianza > conf_etiqueta:
                etiqueta_valida = confianza >= 0.30
                conf_etiqueta   = confianza
                region_etiqueta = region
                coords_etiqueta = [round(c, 1) for c in bbox]

    nodo_valido = gabinete_valido and etiqueta_valida

    if not gabinete_valido and not etiqueta_valida:
        motivo = "No se detecto el gabinete ni la etiqueta del Nodo Safe City"
    elif not gabinete_valido:
        motivo = "No se detecto el gabinete del Nodo Safe City"
    elif not etiqueta_valida:
        motivo = "No se detecto la etiqueta en el Nodo Safe City"
    else:
        motivo = "Gabinete y etiqueta detectados correctamente"

    return JSONResponse({
        "aprobada":          nodo_valido,
        "nodo_valido":       nodo_valido,
        "gabinete_valido":   gabinete_valido,
        "etiqueta_valida":   etiqueta_valida,
        "conf_gabinete":     round(conf_gabinete * 100),
        "conf_etiqueta":     round(conf_etiqueta * 100),
        "region_gabinete":   region_gabinete,
        "region_etiqueta":   region_etiqueta,
        "coords_gabinete":   coords_gabinete,
        "coords_etiqueta":   coords_etiqueta,
        "motivo":            motivo,
        "total_detecciones": len(todas_det),
        "todas_detecciones": todas_det
    })


# ── FO Ingresa al Nodo ────────────────────────────────────────────────────────

@app.post("/fo-nodo/detectar")
async def detectar_fo_nodo(req: DetectarRequest):
    model       = get_model("fo_nodo")
    image       = decode_image(req.image_base64)
    det, clases = run_detection(model, image, req.confianza)
    aprobada    = len(det) > 0
    return JSONResponse({
        "aprobada":         aprobada,
        "total":            len(det),
        "clases":           clases,
        "motivo":           f"FO detectada: {', '.join(clases)}" if aprobada else "No se detecto la fibra optica ingresando al nodo.",
        "confianza_minima": req.confianza,
        "detecciones":      det
    })


# ── Manguera Funda Sellada ────────────────────────────────────────────────────

@app.post("/manguera/detectar")
async def detectar_manguera(req: DetectarRequest):
    model       = get_model("manguera")
    image       = decode_image(req.image_base64)
    det, clases = run_detection(model, image, req.confianza)
    aprobada    = len(det) > 0
    return JSONResponse({
        "aprobada":         aprobada,
        "total":            len(det),
        "clases":           clases,
        "motivo":           f"Manguera funda sellada detectada: {', '.join(clases)}" if aprobada else "No se detecto la manguera funda sellada.",
        "confianza_minima": req.confianza,
        "detecciones":      det
    })


# ── Cable Concentrico ─────────────────────────────────────────────────────────

@app.post("/cable/detectar")
async def detectar_cable(req: DetectarRequest):
    model       = get_model("cable")
    image       = decode_image(req.image_base64)
    det, clases = run_detection(model, image, req.confianza)
    aprobada    = len(det) > 0
    return JSONResponse({
        "aprobada":         aprobada,
        "total":            len(det),
        "clases":           clases,
        "motivo":           f"Cable concentrico detectado: {', '.join(clases)}" if aprobada else "No se detecto el cable concentrico ingresando al nodo.",
        "confianza_minima": req.confianza,
        "detecciones":      det
    })


# ── Roseta Instalada ──────────────────────────────────────────────────────────

@app.post("/roseta/detectar")
async def detectar_roseta(req: DetectarRequest):
    model       = get_model("roseta")
    image       = decode_image(req.image_base64)
    det, clases = run_detection(model, image, req.confianza)
    aprobada    = len(det) > 0

    partes = []
    if "roseta"           in clases: partes.append("roseta instalada")
    if "pigtail_patchcord" in clases: partes.append("pigtail/patchcord")

    return JSONResponse({
        "aprobada":         aprobada,
        "total":            len(det),
        "clases":           clases,
        "motivo":           f"Detectado: {', '.join(partes)}" if aprobada else "No se detecto la roseta ni el pigtail/patchcord.",
        "confianza_minima": req.confianza,
        "detecciones":      det
    })


# ── Ventiladores Nodo (1-4) ───────────────────────────────────────────────────

VENTILADORES_LABELS = {
    1: "Ventilador 1 - Superior",
    2: "Ventilador 2 - Superior",
    3: "Ventilador 3 - Inferior",
    4: "Ventilador 4 - Inferior",
}

@app.post("/ventiladores/detectar")
async def detectar_ventilador(req: VentiladorRequest):
    if req.ventilador_id not in VENTILADORES_LABELS:
        return JSONResponse({"error": "ventilador_id debe ser 1, 2, 3 o 4"}, status_code=400)

    model       = get_model(f"ventilador_{req.ventilador_id}")
    label       = VENTILADORES_LABELS[req.ventilador_id]
    image       = decode_image(req.image_base64)
    det, clases = run_detection(model, image, req.confianza)
    aprobada    = len(det) > 0

    return JSONResponse({
        "aprobada":         aprobada,
        "ventilador_id":    req.ventilador_id,
        "ventilador_label": label,
        "total":            len(det),
        "clases":           clases,
        "motivo":           f"{label} detectado: {', '.join(clases)}" if aprobada else f"No se detecto el {label}.",
        "confianza_minima": req.confianza,
        "detecciones":      det
    })


# ── UPS de Respaldo ───────────────────────────────────────────────────────────

@app.post("/ups/detectar")
async def detectar_ups(req: DetectarRequest):
    model       = get_model("ups")
    image       = decode_image(req.image_base64)
    det, clases = run_detection(model, image, req.confianza)
    aprobada    = len(det) > 0
    return JSONResponse({
        "aprobada":         aprobada,
        "total":            len(det),
        "clases":           clases,
        "motivo":           f"UPS de respaldo detectado: {', '.join(clases)}" if aprobada else "No se detecto el UPS de respaldo.",
        "confianza_minima": req.confianza,
        "detecciones":      det
    })


# ── Bateria ───────────────────────────────────────────────────────────────────

@app.post("/bateria/detectar")
async def detectar_bateria(req: DetectarRequest):
    model       = get_model("bateria")
    image       = decode_image(req.image_base64)
    det, clases = run_detection(model, image, req.confianza)
    aprobada    = len(det) > 0
    return JSONResponse({
        "aprobada":         aprobada,
        "total":            len(det),
        "clases":           clases,
        "motivo":           f"Bateria detectada: {', '.join(clases)}" if aprobada else "No se detecto la bateria.",
        "confianza_minima": req.confianza,
        "detecciones":      det
    })
