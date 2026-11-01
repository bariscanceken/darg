# DARG mobil uygulaması (Flutter)

DARG asistanının Android/iOS uygulaması. Ana proje: [../README.md](../README.md).

- **Sohbet:** cümle cümle kaynak işaretleri, kaynaktan belgeyi ilgili sayfada açma, 👍/👎 geri bildirim, sohbet geçmişi
- **Kütüphane:** 139 AFAD belgesi, belge indirme, internetsiz kelime araması
- **Yardım sinyali:** yüksek sesli düdük, SOS feneri, ekranla SOS
- **Acil durum kartı:** telefonda şifreli saklanır, sunucuya gönderilmez; isteğe bağlı Android kilit ekranı bildirimi
- **Duyurular:** doğrulanmış kurumların yönetici onaylı duyuruları; kurum başvurusu ve uygulama içi yönetim

```bash
flutter pub get
flutter test
flutter run --dart-define=DARG_API=https://sunucu-adresiniz   # ya da uygulamada Hakkında → Sunucu adresi
flutter build apk --release --dart-define=DARG_API=https://sunucu-adresiniz
```

Düdük sesi `tool/make_whistle.py` ile üretilir (`assets/sounds/whistle.wav`).
