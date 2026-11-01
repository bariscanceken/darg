# DARG'a katkı

**Türkçe** · [English](#english)

Katkınız için teşekkürler. DARG insanların deprem anında ve sonrasında başvurabileceği bir bilgi asistanı; bu yüzden
burada "çalışıyor gibi görünmesi" yetmez, **ölçülmüş olması** gerekir. Aşağıdaki kurallar bunun için var.

## Nasıl katkı verebilirsiniz?

- **Hata bildirimi (Issue):** sorulan soruyu, alınan cevabı (ya da ekran görüntüsünü), beklediğiniz cevabı ve
  mümkünse kaynağı (AFAD belgesi, sayfa) yazın. Yanlış ya da tehlikeli olabilecek bir cevap gördüyseniz başlığa
  **[güvenlik]** ekleyin; bunlara öncelik verilir.
- **Öneri:** yeni özellik ya da büyük değişiklik için önce bir Issue açıp yaklaşımı konuşalım, sonra kod.
- **Kod (Pull Request):** küçük ve tek konulu PR'lar tercih edilir.
- **Veri:** eksik ya da güncellenmiş AFAD yayınlarını, yanlış eşlenmiş il/ilçe adlarını bildirmek de değerli bir katkıdır.

## Geliştirme ortamı

```bash
pip install -r requirements.txt
cp .env.example .env            # yalnız gerçek soru sormak için OPENAI_API_KEY gerekir
```

AFAD belgeleri depoda yok; indirme ve index üretimi için [docs/VERI.md](docs/VERI.md). Testlerin çoğu belgeler ve
API anahtarı olmadan çalışır.

## Testler

```bash
python -m unittest tests.test_pipeline_v3 tests.test_graph_v3 tests.test_api tests.test_library tests.test_community
cd mobile && flutter test
```

Testler **sahte bir OpenAI istemcisiyle** çalışır, ücret doğurmaz. Yeni bir davranış ekliyorsanız testini de ekleyin;
OpenAI'ye gerçekten istek atan test yazmayın. (`python -m unittest discover` bu klasör düzeninde çalışmıyor; modülleri
adıyla verin.)

## Ölçüm kuralları (en önemli kısım)

1. **Ölçmeden benimseme yok.** Cevapları etkileyen her değişiklik (prompt, model, arama, eşik) önceden yazılmış bir soru
   setiyle ölçülerek gelmeli. PR açıklamasına önce/sonra sonuçlarını ve kullanılan seti yazın.
2. **Geliştirme ve bağımsız setler ayrıdır.** Ayar yaparken geliştirme setlerini kullanın. Bağımsız bir set, sistem
   onu görmeden yazılır, bir kez çalıştırılır ve sonucu görüldükten sonra eşik değiştirilmez.
3. **Kapalı test setleri:** `data/evaluation/final_test_v1*` okunmaz, değiştirilmez, çalıştırılmaz. Final Test v2 ve
   `model_check_v1` birer kez çalıştırıldı; betikleri yeniden çalışmayı reddeder. Bu dosyalara dokunan PR'lar kabul
   edilmez.
4. **Soruya özel yama yerine genel çözüm.** "Şu soru yanlış çıkıyor" diye o soruya özel kural eklemek yerine kök nedeni
   (arama, durum eşleşmesi, kaynak türü…) düzeltin.
5. Kütüphane varsayılanları (`generate_answer()`) Final Test v1 yolunu korur; yeni davranışlar isteğe bağlı açılır
   (CLI bayrağı ya da v3 hattı).

## Güvenlik kuralları

Cevap hattına dokunan değişiklikler şunları bozmamalı (testleri var):

- Belirli bir bina için "güvenlidir" güvencesi verilmez.
- Deprem tahmini yapılmaz.
- Acil durum sorularında cevap 112 ile başlar.
- AFAD'a dayanmayan cümle "(genel bilgi)" olarak işaretlenir; alıntılar kodla doğrulanır.
- Web araması yalnız resmî kaynaklarla (gov.tr, kizilay.org.tr) sınırlıdır.

## Gizli bilgiler ve kişisel veri

- API anahtarı, şifre ya da sunucu bilgisi koda yazılmaz; `.env` git'e girmez (şablon `.env.example`).
- Ölçüm çıktılarında belgelerden gelen kişisel bilgi (telefon, e-posta, isim) olabilir; böyle dosyaları commit etmeden
  önce kontrol edin.
- Kullanıcı kayıtları (`data/logs/`) ve uygulama veritabanı (`data/app/`) depoya eklenmez.

## Kod ve commit stili

- Çevredeki kodun stiline uyun: isimlendirme, yorum yoğunluğu, Türkçe kullanıcı metinleri, İngilizce kod/yorum.
- Commit başlığı: küçük harfle başlayan, noktayla biten İngilizce cümle; gövdede madde listesi. Örnek:
  ```
  match first-aid advice to the injury type.

  - heavy-object and crush situations no longer retrieve sprain advice
  - new test cases in tests/test_pipeline_v3.py
  ```
- PR açıklaması: ne değişti, neden, nasıl test edildi, (cevapları etkiliyorsa) ölçüm sonucu.

## Lisans

Gönderdiğiniz katkılar deponun [MIT lisansı](LICENSE) altında yayımlanır. AFAD belgeleri lisans kapsamı dışındadır;
belge dosyalarını PR'a eklemeyin.

---

<a id="english"></a>
## English

Thanks for contributing. DARG is meant to be used during and after earthquakes, so changes must be **measured**, not
just look right.

- **Issues:** include the question, the answer you got, the answer you expected and, if possible, the source (AFAD
  document and page). Tag possibly dangerous answers with **[security]**. Discuss larger changes in an issue first.
- **Setup:** `pip install -r requirements.txt`, copy `.env.example` to `.env`. AFAD documents are not in the repo, see
  `docs/VERI.md`.
- **Tests:** `python -m unittest tests.test_pipeline_v3 tests.test_graph_v3 tests.test_api tests.test_library
  tests.test_community` and `cd mobile && flutter test`. Tests use a fake OpenAI client; never add tests that call the
  real API.
- **Measure before adopting:** any change that affects answers comes with before/after results on a question set
  written beforehand. Development and independent sets are kept apart. Final Test v1 is never read, changed or run;
  Final Test v2 and `model_check_v1` were run once and stay closed. Prefer general fixes over question-specific patches.
- **Safety rules must hold:** no guarantee about a specific building, no earthquake prediction, emergency answers start
  with 112, non-AFAD sentences are labelled "(genel bilgi)", web search only on official domains.
- **Secrets and personal data:** never commit API keys, `.env`, logs or the app database; check evaluation dumps for
  personal data quoted from documents.
- **Commits:** lowercase English subject ending with a period, bullet-list body.
- Contributions are released under the [MIT license](LICENSE); do not add AFAD document files.
