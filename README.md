# DARG – Deprem Asistanı

**Türkçe** · [English](#english)

DARG, depremle ilgili sorulara **AFAD belgelerine dayanarak**, her cümlenin kaynağını göstererek cevap veren bir yapay
zekâ asistanıdır. Afet sonrasında yayılan yanlış bilgiye karşı, kaynağı belli ve güvenilir bilgi sunmayı amaçlar.

TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projesi · Karadeniz Teknik Üniversitesi, Yönetim Bilişim Sistemleri ·
Proje yürütücüsü: Barışcan Çeken · Danışman: Dr. Öğr. Üyesi Murat Kılınç · Web: [darg.network](https://darg.network)

> DARG bir bilgi asistanıdır; 112, AFAD ya da sağlık profesyonellerinin yerini tutmaz. Acil bir durumda önce 112'yi arayın.

## Neler yapıyor?

- **Kaynaklı cevap:** her cümle ya bir AFAD belgesinden birebir alıntıyla desteklenir (alıntı kodla doğrulanır, belge
  ve sayfa gösterilir) ya da açıkça "(genel bilgi)" diye işaretlenir.
- **Hibrit karar ajanı:** AFAD belgelerinde dayanak yoksa ya da güncel bilgi isteniyorsa (hastane, adres, tarih…) yalnızca
  resmî sitelerde (gov.tr, Kızılay) web araması yapılır; web kaynakları tarih uyarısıyla gösterilir.
- **Güvenlik kuralları:** belirli bir bina için güvence verilmez, deprem tahmini yapılmaz, acil durum sorularında cevap
  "112'yi arayın" ile başlar, konu dışı sorular kibarca reddedilir.
- **Bölgesel bilgi:** il/ilçe adı yazım hatalarına dayanıklı tanınır ("antakyadayım", "samandagda"); ilin afet planı
  önceliklendirilir.
- **Mobil uygulama (Flutter):** sohbet, internetsiz okunabilen 139 belgelik kütüphane, yardım sinyali (düdük, SOS feneri),
  telefonda şifreli saklanan acil durum kartı, doğrulanmış kurumların yönetici onaylı duyuruları.

## Mimari

```
Soru → soru anlama (yazım, tip, durum, 3 arama sorgusu) → konu kontrolü → yer tespiti
     → geniş arama: 3 sorgu × (E5 vektör + BM25) + il planı + kaynak türü havuzu → cross-encoder → ilk 8 parça
     → cevap (her cümle: AFAD alıntısı ya da "genel bilgi") → alıntı doğrulama
     → hibrit karar: güncel bilgi / dayanak yok → resmî web araması → cevap yeniden yazılır
```

Orkestrasyon **LangGraph** ile (`src/rag/graph_v3.py`). Belgeler yapı korunarak işlenir (başlık seviyeleri, tablolar,
"Belge › Bölüm › Alt bölüm" başlıklı ~31 bin parça); görsel sayfalar OCR ile okunur.

| Klasör | İçerik |
|---|---|
| `src/preprocessing/` | PDF çıkarımı, OCR, parçalama, il-ilçe sözlüğü, il kartları |
| `src/retrieval/` | E5-small FAISS index, BM25, yeniden sıralama |
| `src/rag/` | cevap hatları (v3 varsayılan), konuşma hafızası, yer tespiti |
| `src/api/` | FastAPI: soru-cevap, geri bildirim, istatistik, kütüphane, kurum hesapları ve duyurular |
| `src/evaluation/` | ölçüm betikleri (sonuçlar `data/evaluation/`) |
| `mobile/` · `website/` · `deploy/` | Flutter uygulaması · web sitesi · Docker, Caddy (HTTPS), kurulum rehberi |

## Ölçüm sonuçları

| Ölçüm | Sonuç |
|---|---|
| **Final Test v2** (42 bağımsız soru, tek sefer, **uzman değerlendirmesi**, cevap modeli gpt-4o) | **40/42 doğru (%95,2)**, tehlikeli tavsiye 0 |
| Bağımsız model kontrolü (32 yeni soru, tek sefer, cevap modeli gpt-4.1-mini, Claude puanladı) | 29/32 doğru (%90,6), yanlış 1 |
| Bölgesel arama (ilk 5 sonuçta yanlış ilin belgesi) | 40 → 0 |
| Arama başarısı, parçalama düzeltmesi sonrası (holdout, Hit@5) | 20/30 → 26/30 |

Projenin hedefi en az %85 doğruluktu. Ayrıntılı geliştirme ve ölçüm raporu: `docs/darg_calisma_raporu.html`.

## Kurulum

Gereksinimler: Python 3.12+, bir OpenAI API anahtarı, AFAD belgeleri (depoda yok, bkz. [docs/VERI.md](docs/VERI.md)).

```bash
pip install -r requirements.txt
cp .env.example .env          # OPENAI_API_KEY=... yaz
# AFAD belgelerini data/raw/documents/afad/ altına koy ve index'i üret (docs/VERI.md)
python -m src.rag.generate_answer --chat            # terminalde sohbet
uvicorn src.api.app:app --port 8000                  # API; belgeler: http://localhost:8000/docs
```

Sunucuya kurulum (Ubuntu + Docker, otomatik HTTPS, tek komutla güncelleme): [deploy/KURULUM.md](deploy/KURULUM.md).
Mobil uygulama: `cd mobile && flutter run` (sunucu adresi uygulamanın Hakkında ekranından girilir).

Maliyet: cevap başına yaklaşık 0,003 $ (gpt-4.1-mini); web araması gereken sorularda ~0,06 $.

## Testler

```bash
python -m unittest tests.test_pipeline_v3 tests.test_graph_v3 tests.test_api tests.test_library tests.test_community
cd mobile && flutter test
```
Testler sahte bir OpenAI istemcisiyle çalışır; ücret doğmaz.

## Katkı

Hata bildirimi ve önerilere açığız: GitHub Issues üzerinden yazabilirsiniz. Değişiklik önerirken lütfen ölçüm
kuralına uyun: davranışı değiştiren her öneri, önceden yazılmış bir soru setiyle ölçülerek gelmeli. Ayrıntılar:
[CONTRIBUTING.md](CONTRIBUTING.md).

## Lisans

Kod ve belgeler [MIT](LICENSE) lisanslıdır. AFAD belgeleri depoda değildir ve lisans kapsamı dışındadır.

---

<a id="english"></a>
## English

DARG is a Turkish earthquake information assistant. It answers questions **from the documents of AFAD** (Türkiye's
Disaster and Emergency Management Authority) and shows the basis of every sentence: a verbatim AFAD quote (checked in
code, with document and page) or an explicit "general knowledge" label. When the documents have no basis or the
question needs current information, a hybrid decision agent searches official web sources only (gov.tr, Turkish Red
Crescent). Safety rules: no guarantees about a specific building, no earthquake prediction, emergency answers start
with "call 112", off-topic questions are declined.

Stack: structure-aware PDF processing with OCR, multilingual E5 embeddings + BM25 + cross-encoder reranking,
LangGraph orchestration, OpenAI models (gpt-4.1-mini answers, gpt-4o web search), FastAPI, a Flutter mobile app
(chat, offline document library, help signal, emergency card, moderated announcements from verified organisations),
Docker + Caddy deployment.

Results: on a held-out 42-question test graded by a domain expert the system answered **40/42 (95.2%)** correctly with
no dangerous advice (target: ≥ 85%). A later one-shot check of the cheaper answer model scored 29/32 (90.6%).

Built as a TÜBİTAK 2209-A undergraduate research project at Karadeniz Technical University (Management Information
Systems). AFAD documents are not included (see `docs/VERI.md`). Code: MIT license. DARG gives information only and does
not replace emergency services.
