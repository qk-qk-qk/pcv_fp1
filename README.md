# Scrolly Bird 🐦📸

**Scrolly Bird** adalah game bergaya *arcade* di mana pemain harus menghindari rintangan pipa (kotak) yang terus berjalan. 

Keunikan utama dari game ini adalah **sistem kontrol berbasis Computer Vision (OpenCV)**. Anda tidak perlu menggunakan keyboard; cukup gerakkan benda berwarna cerah (seperti sarung tangan) ke atas dan ke bawah di depan *webcam*, dan karakter burung akan mengikuti pergerakan tangan Anda secara *real-time*!

---

## 🎮 Fitur Game
*   **Computer Vision Control:** Mengontrol pergerakan karakter menggunakan deteksi warna objek (HSV Tracking) melalui kamera.
*   **Calibration Tool (`--calibrate`):** Dilengkapi dengan alat kalibrasi bawaan berbentuk *slider* GUI untuk menyesuaikan sensitivitas warna dengan kondisi pencahayaan ruangan Anda, lalu menyimpannya secara otomatis ke `hsv.json`.
*   **Debug Mode:** Jendela *Picture-in-Picture* OpenCV yang bisa dimunculkan/disembunyikan saat game berjalan (dengan menekan tombol `D`) untuk melihat apa yang sedang dideteksi oleh kamera.
*   **Mouse Wheel Fallback:** Jika kamera tidak tersedia, game tetap bisa dimainkan menggunakan *scroll* pada mouse.
*   **Multi-threading:** Pemrosesan gambar kamera dan *rendering* game dipisah ke dalam *thread* yang berbeda agar FPS game tetap mulus (60 FPS) tanpa terhambat (lag) oleh proses OpenCV.

---

## 🎥 Video Demo

[![Scrolly Bird Demo](https://img.youtube.com/vi/0tVBzbhoLZs.jpg)](https://www.youtube.com/watch?v=0tVBzbhoLZs)

---

## 🚀 Cara Main & Instalasi

### 1. Kebutuhan Sistem (Prerequisites)
Pastikan Anda sudah menginstal library Python yang dibutuhkan. Buka terminal/CMD dan jalankan:
```bash
pip install raylib opencv-python numpy
```

### 2. Struktur Folder
Pastikan file gambar/aset diletakkan di dalam folder `pictures/` yang berada di direktori yang sama dengan `scrolly_bird.py`:
```text
/scrolly_bird.py
/hsv.json (akan dibuat otomatis setelah kalibrasi)
/pictures/
  ├── background.png
  ├── bird.png
  ├── abv boxes scrolly.png
  └── blw boxes scrolly.png
```

### 3. Menjalankan Game
*   **Untuk bermain langsung:**
    ```bash
    python scrolly_bird.py
    ```
*   **Untuk kalibrasi warna (jika objek tidak terdeteksi dengan baik):**
    ```bash
    python scrolly_bird.py --calibrate
    ```
    *(Geser slider hingga objek Anda berwarna putih utuh di layar hitam, lalu tekan `S` untuk menyimpan).*

### 4. Kontrol Game
*   **Kamera (Utama):** Gerakkan objek dengan warna yang telah dikalibrasi naik/turun di depan layar.
*   **Mouse Wheel:** Putar ke atas/bawah sebagai kontrol cadangan.
*   **Enter:** Mengulang (Retry) saat Game Over.
*   **Tombol `D`:** Menampilkan/menyembunyikan jendela kamera (Debug HSV) saat game berjalan.

---

## ⚙️ Filter Pemrosesan Gambar (Computer Vision)
Agar kontrol karakter burung tidak melompat-lompat dan kamera hanya fokus pada sarung tangan Anda, kode ini mengimplementasikan sebuah *pipeline* pemrosesan gambar (*Computer Vision*) yang terdiri dari beberapa filter berurutan. 

Berikut adalah filter yang digunakan dan **mengapa** masing-masing filter tersebut digunakan:

### 1. Filter Warna HSV (Hue, Saturation, Value Masking)
*   **Apa yang dilakukan:** Menggunakan `cv2.inRange()`, gambar dari kamera dikonversi dari BGR (merah-hijau-biru biasa) ke format HSV. Sistem membuang (menghitamkan) semua piksel yang warnanya berada di luar nilai target.
*   **Mengapa digunakan:** 
    *   Format warna RGB sangat sensitif terhadap perubahan cahaya. Format HSV memisahkan informasi warna asli (*Hue*) dari kecerahan (*Value*). 
    *   Dengan mengatur nilai *Saturation* cukup tinggi sebagai ambang batas, kita bisa mencegah wajah atau lengan pemain (yang warnanya mungkin sepintas mirip namun saturasinya rendah/pudar) ikut terdeteksi oleh sistem sebagai input kontrol.

### 2. Filter Morfologi Noise (Erosion & Dilation)
*   **Apa yang dilakukan:** Sistem menjalankan fungsi `cv2.erode` (mengikis area putih) lalu dilanjutkan dengan `cv2.dilate` (menebalkan area putih).
*   **Mengapa digunakan:** 
    *   Kamera webcam sering kali menangkap *noise* (bintik-bintik kecil piksel acak yang warnanya menyerupai sarung tangan). *Erode* secara efektif menghapus bintik-bintik pengganggu ini.
    *   Karena sarung tangan yang digunakan memiliki motif/bintik warna putih, hasil filter HSV sering kali berlubang. *Dilate* bertugas menambal lubang-lubang tersebut sehingga tangan Anda tetap terdeteksi sebagai satu bentuk *(shape)* yang utuh dan solid.

### 3. Filter Luas Area Kontur (Minimum Contour Area)
*   **Apa yang dilakukan:** Menggunakan `cv2.contourArea(c) > MIN_AREA`, program menghitung luas piksel dari bercak warna yang lolos filter, lalu membuang bercak yang luasnya kurang dari 150 piksel.
*   **Mengapa digunakan:** Di latar belakang ruangan mungkin ada benda kecil atau pantulan cahaya yang kebetulan memiliki warna yang persis sama dengan sarung tangan. Filter ini memastikan sistem "mengabaikan" benda-benda kecil (sampah visual) tersebut dan hanya bereaksi pada objek berukuran dominan di depan kamera (tangan pemain).

### 4. Filter Kestabilan Posisi (Bounding Box & Center Extraction)
*   **Apa yang dilakukan:** Menggunakan `cv2.boundingRect(c)`, sistem menggambar kotak lurus imajiner yang membungkus sempurna sarung tangan, lalu secara matematis mengambil nilai titik tengah sumbu Y (ketinggian) dari kotak tersebut.
*   **Mengapa digunakan:** Bentuk tangan yang sedang digerak-gerakkan memiliki wujud yang tidak beraturan dan titik berat yang terus bergeser secara ekstrem (misal ketika jari terbuka vs mengepal). Jika tidak dibungkus kotak pembatas, posisi burung di dalam game akan ikut gemetar/bergetar (*jittering*). Titik tengah dari kotak pembatas memberikan nilai koordinat Y yang jauh lebih stabil dan mulus untuk dimainkan.