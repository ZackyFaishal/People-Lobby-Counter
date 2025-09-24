# 🧑‍🤝‍🧑 People Lobby Counter

## 📄 Deskripsi Proyek
Proyek **People Lobby Counter** ini bertujuan untuk menghitung jumlah orang **masuk (IN)** dan **keluar (OUT)** di area lobby secara **real-time** menggunakan:
- **YOLOv11** untuk deteksi orang
- **ByteTrack** untuk pelacakan (tracking) orang
- **OpenCV** untuk pengolahan video dan visualisasi hasil  

Aplikasi ini menampilkan **jumlah orang yang masuk dan keluar** secara langsung pada video, beserta **FPS (Frame per Second)** untuk memantau performa sistem.  
Output dapat ditampilkan langsung di layar atau disimpan sebagai video hasil deteksi.

---

## 🚀 Fitur Utama
- **Real-Time People Counting:** Menghitung orang yang masuk/keluar pada area lobby.
- **Polygon-Based Counting Zone:** Menggunakan poligon khusus untuk area masuk dan keluar.
- **YOLOv11 + ByteTrack:** Akurat dan cepat untuk deteksi serta pelacakan objek.
- **FPS Monitoring:** Menampilkan performa sistem secara langsung.
- **Save Result Option:** Menyimpan video hasil deteksi beserta anotasi.

---

## 📊 Cara Kerja
1. Sistem mendeteksi orang menggunakan model **YOLOv11**.
2. Menggunakan **ByteTrack** untuk melacak ID setiap orang agar tidak dihitung lebih dari sekali.
3. Dua area poligon ditetapkan:
   - **Entry Area (IN)** – menghitung orang yang masuk.
   - **Exit Area (OUT)** – menghitung orang yang keluar.
4. Menghitung jumlah orang berdasarkan titik pusat bounding box di area poligon.
5. Menampilkan hasil dalam jendela fullscreen, menampilkan **counter** dan **FPS**.

---
