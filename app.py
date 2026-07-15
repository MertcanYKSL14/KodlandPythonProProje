import os
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, session
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

from questions import sorular

app = Flask(__name__)
app.secret_key = "super-secret-key-987"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sinav.db"
db = SQLAlchemy(app)

UPLOAD_FOLDER = os.path.join("static", "uploads")
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


goruntu_modeli = load_model("model/keras_model.h5", compile=False)

sinif_isimleri = []
with open("model/labels.txt", "r", encoding="utf-8") as f:
    for satir in f:
        satir = satir.strip()
        if satir:
            parca = satir.split(" ", 1)
            isim = parca[1] if len(parca) > 1 else parca[0]
            sinif_isimleri.append(isim)


def gorseli_tahmin_et(dosya_yolu):
    veri = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    resim = Image.open(dosya_yolu).convert("RGB")
    resim = ImageOps.fit(resim, (224, 224), Image.Resampling.LANCZOS)
    resim_dizisi = np.asarray(resim)
    normalize_resim = (resim_dizisi.astype(np.float32) / 127.5) - 1
    veri[0] = normalize_resim

    tahmin = goruntu_modeli.predict(veri)
    index = int(np.argmax(tahmin))
    sinif_adi = sinif_isimleri[index]
    guven = float(tahmin[0][index])
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