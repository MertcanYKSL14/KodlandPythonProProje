import os
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, session
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from questions import sorular

app = Flask(__name__)
app.secret_key = "super-secret-key-987"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sinav.db"
db = SQLAlchemy(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


class Attempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    isim = db.Column(db.String(100), nullable=False)
    skor = db.Column(db.Integer, nullable=False)
    tarih = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Attempt {self.isim} - %{self.skor}>"


class ImagePrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dosya_adi = db.Column(db.String(200), nullable=False)
    tahmin_sinifi = db.Column(db.String(100), nullable=False)
    guven_skoru = db.Column(db.Float, nullable=False)
    tarih = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ImagePrediction {self.tahmin_sinifi}>"


model_path = os.path.join(BASE_DIR, "model", "model.tflite")
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels_path = os.path.join(BASE_DIR, "model", "labels.txt")
sinif_isimleri = []
with open(labels_path, "r", encoding="utf-8") as f:
    for satir in f:
        satir = satir.strip()
        if satir:
            parca = satir.split(" ", 1)
            isim = parca[1] if len(parca) > 1 else parca[0]
            sinif_isimleri.append(isim)


def gorseli_tahmin_et(dosya_yolu):
    resim = Image.open(dosya_yolu).convert("RGB")
    resim = ImageOps.fit(resim, (224, 224), Image.Resampling.LANCZOS)
    resim_dizisi = np.asarray(resim, dtype=np.float32)
    normalize_resim = (resim_dizisi / 127.5) - 1.0
    input_data = np.expand_dims(normalize_resim, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    index = int(np.argmax(output_data[0]))
    sinif_adi = sinif_isimleri[index]
    guven = float(output_data[0][index])
    return sinif_adi, guven


@app.context_processor
def skor_bilgisi():
    isim = session.get("isim")
    kisisel_en_yuksek = None
    if isim:
        kisisel_en_yuksek = (
            db.session.query(db.func.max(Attempt.skor))
            .filter(Attempt.isim == isim)
            .scalar()
        )
    genel_en_yuksek = db.session.query(db.func.max(Attempt.skor)).scalar()
    return dict(kisisel_en_yuksek=kisisel_en_yuksek, genel_en_yuksek=genel_en_yuksek)


@app.route("/")
def anasayfa():
    return render_template("index.html")


@app.route("/sinav", methods=["GET", "POST"])
def sinav():
    if request.method == "POST":
        isim = request.form.get("isim")
        session["isim"] = isim

        dogru_sayisi = 0
        for soru in sorular:
            secilen = request.form.get(f"soru_{soru['id']}")
            if secilen is not None and int(secilen) == soru["dogru"]:
                dogru_sayisi += 1

        skor = round((dogru_sayisi / len(sorular)) * 100)

        yeni_deneme = Attempt(isim=isim, skor=skor)
        db.session.add(yeni_deneme)
        db.session.commit()

        return render_template("sonuc.html", skor=skor)

    return render_template("sinav.html", sorular=sorular)


@app.route("/gorsel-tanima", methods=["GET", "POST"])
def gorsel_tanima():
    sonuc = None
    if request.method == "POST":
        dosya = request.files.get("gorsel")
        if dosya and dosya.filename != "":
            guvenli_ad = secure_filename(dosya.filename)
            benzersiz_ad = f"{int(datetime.utcnow().timestamp())}_{guvenli_ad}"
            dosya_yolu = os.path.join(app.config["UPLOAD_FOLDER"], benzersiz_ad)
            dosya.save(dosya_yolu)

            sinif_adi, guven = gorseli_tahmin_et(dosya_yolu)

            kayit = ImagePrediction(
                dosya_adi=benzersiz_ad,
                tahmin_sinifi=sinif_adi,
                guven_skoru=guven,
            )
            db.session.add(kayit)
            db.session.commit()

            sonuc = {
                "dosya_adi": benzersiz_ad,
                "sinif_adi": sinif_adi,
                "guven": round(guven * 100, 1),
            }
    return render_template("gorsel_tanima.html", sonuc=sonuc)


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)