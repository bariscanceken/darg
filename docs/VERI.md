# Veri: AFAD belgeleri

DARG'ın bilgi kaynağı, AFAD'ın (Afet ve Acil Durum Yönetimi Başkanlığı) kamuya açık yayınlarıdır. Belgeler bu depoda
**yoktur** (139 dosya, ~2.7 GB): telif hakları yayıncılarına aittir ve depo lisansının (MIT)
kapsamı dışındadır. Sistemi kendi bilgisayarınızda kurmak için belgeleri AFAD'ın web sitesinden (afad.gov.tr: Yayınlar,
Kitaplar, İl Afet Risk Azaltma Planları ve ulusal planlar sayfaları) indirip aşağıdaki klasörlere koyun.

## Klasör düzeni

```
data/raw/documents/afad/
  books/            kitaplar ve rehberler (ör. ilk72saat1.pdf)
  publications/     broşür, afiş, yayınlar
  plans/irap/       il afet risk azaltma planları (IRAP)
  plans/tamp|tarap|tasip|udsep/   ulusal planlar
  infographics/ maps/ statistics/  görseller, haritalar, istatistikler
```

## Veri hattı

```
python -m src.preprocessing.extract_structured      # PDF -> data/processed/structured_v5 (yapı korumalı, ~1 saat)
python -m src.preprocessing.ocr_pages --cache-only  # görsel sayfaların OCR'ı (önbellek yoksa gpt-4o ile, ücretli)
python -m src.preprocessing.build_document_titles
python -m src.preprocessing.chunk_structured_v5     # ~31 bin parça
python -m src.retrieval.build_index_v2 --chunks-file data/processed/chunks/afad_earthquake_chunks_v5.jsonl \
       --index-dir data/indexes/mobile_v5
python -m src.preprocessing.build_gazetteer         # il-ilçe ve il planı eşlemesi (data/geo/, depoda var)
```

Depoda bulunan küçük veriler: `data/geo/` (il-ilçe listesi, belge başlıkları ve türleri), `data/province_cards/`
(il planlarından derlenmiş doğrulanmış il kartları), `data/query_normalization/` (yazım düzeltme sözlüğü) ve
`data/evaluation/` (soru setleri ve ölçüm sonuçları).

## Belge listesi

### Vatandaş rehberleri (21)

| Başlık | Dosya | Klasör |
|---|---|---|
| Afet Barınma Faaliyetleri | `Afet-barinma-faaliyetleri.pdf` | `afad/publications` |
| Afet Türlerine Göre Yapısal Olmayan Risk Azaltma | `document-_11_.pdf` | `afad/books` |
| Afet Çantası Bilgilendirme Afişi | `13_afad_bilgilendirme_afisleri_tr_afet_cantasi_info_.pdf` | `afad/publications` |
| Afetlerde İlk 72 Saat Hazırlık Rehberi | `ilk72saat1.pdf` | `afad/books` |
| AYDES: Afet Yönetimi ve Karar Destek Sistemi | `AYDES.pdf` | `afad/publications` |
| Bilgilen, Planla, Hazırlan | `bilgilenplanlahazirlan.pdf` | `afad/publications` |
| Deprem Nedir? | `deprem_nedir.pdf` | `afad/publications` |
| Engelliler için Afet Hazırlık Rehberi | `ilk72saatengelliler.pdf` | `afad/books` |
| Hasar Tespit Broşürü | `hasar_tespit_-_10x21.pdf` | `afad/publications` |
| İkaz ve Alarm İşaretleri | `04_ikaz_alarm_isaretleri_tr-REVIZE.pdf` | `afad/publications` |
| İlk 72 Saat Afet Hazırlık Yayını | `ilk72saat.pdf` | `afad/publications` |
| İlk 72 Saat: Yetişkinler İçin | `ilk72saatyetiskin.pdf` | `afad/publications` |
| İlk 72 Saat: Çocuklar İçin | `ilk72saatcocuk.pdf` | `afad/publications` |
| Olağandışı Durumlar ve Hayatın İdamesi | `olagandisi.pdf` | `afad/books` |
| Psikolojik İlk Yardım | `psikolojikilkyardim.pdf` | `afad/books` |
| Psikososyal Destek Broşürü | `psikososyal_-_10x21.pdf` | `afad/publications` |
| Soru-Cevap Kitapçığı | `SoruCevap-Kitapcik-2.pdf` | `afad/publications` |
| Toplanma Alanları | `toplanma_alanlari.pdf` | `afad/publications` |
| Yapılacak Yardımlar Broşürü | `yapilacak_yardimlar_-_10x21.pdf` | `afad/publications` |
| Yapısal Olmayan Risklerin Azaltılması | `yapisalolmayanriskler.pdf` | `afad/books` |
| Yer Seçimi | `yer_secimi.pdf` | `afad/publications` |

### İl afet planları (69)

| Başlık | Dosya | Klasör |
|---|---|---|
| Adana İl Afet Risk Azaltma Planı | `ADANA-IRAP.pdf` | `afad/plans/irap` |
| Adıyaman İl Afet Risk Azaltma Planı | `adiyaman-irap.pdf` | `afad/plans/irap` |
| Aksaray İl Afet Risk Azaltma Planı | `AKSARAY-IRAP-2021.pdf` | `afad/plans/irap` |
| Ankara İl Afet Risk Azaltma Planı | `ANKARA_IRAP_2021.pdf` | `afad/plans/irap` |
| Antalya İl Afet Risk Azaltma Planı | `Antalya-IRAP_2022.pdf` | `afad/plans/irap` |
| Ardahan İl Afet Risk Azaltma Planı | `Ardahan-IRAP.pdf` | `afad/plans/irap` |
| Artvin İl Afet Risk Azaltma Planı | `ARTVIN-IRAP-SON_22102021.pdf` | `afad/plans/irap` |
| Aydın İl Afet Risk Azaltma Planı | `Aydin-IRAP-2021.pdf` | `afad/plans/irap` |
| Ağrı İl Afet Risk Azaltma Planı | `IRAP_AGRI.pdf` | `afad/plans/irap` |
| Balıkesir İl Afet Risk Azaltma Planı | `BALIKESIR-IRAP.pdf` | `afad/plans/irap` |
| Bartın İl Afet Risk Azaltma Planı | `BARTIN-IRAP-2022.pdf` | `afad/plans/irap` |
| Batman İl Afet Risk Azaltma Planı | `Batman-IRAP.pdf` | `afad/plans/irap` |
| Bayburt İl Afet Risk Azaltma Planı | `Bayburt-IRAP.pdf` | `afad/plans/irap` |
| Bilecik İl Afet Risk Azaltma Planı | `bilecik_irap_2023.pdf` | `afad/plans/irap` |
| Bingöl İl Afet Risk Azaltma Planı | `BINGOL-AFAD-IRAP-2012.pdf` | `afad/plans/irap` |
| Bitlis İl Afet Risk Azaltma Planı | `BITLIS_IRAP.pdf` | `afad/plans/irap` |
| Bolu İl Afet Risk Azaltma Planı | `Bolu_irap_01_11_2021.pdf` | `afad/plans/irap` |
| Burdur İl Afet Risk Azaltma Planı | `Burdur-irap.pdf` | `afad/plans/irap` |
| Bursa İl Afet Risk Azaltma Planı | `Bursa-IRAP.pdf` | `afad/plans/irap` |
| Denizli İl Afet Risk Azaltma Planı | `Denizli_Il_Afet_Risk_Azaltma-Plani.pdf` | `afad/plans/irap` |
| Diyarbakır İl Afet Risk Azaltma Planı | `Diyarbakir-IRAP-kitapcik.pdf` | `afad/plans/irap` |
| Elazığ İl Afet Risk Azaltma Planı | `Elazig_irap__3103-_1_-yeni.pdf` | `afad/plans/irap` |
| Erzincan İl Afet Risk Azaltma Planı | `ERZINCAN_IRAP_2021-.pdf` | `afad/plans/irap` |
| Eskişehir İl Afet Risk Azaltma Planı | `ESKISEHIR-IRAP.pdf` | `afad/plans/irap` |
| Gaziantep İl Afet Risk Azaltma Planı | `Gaziantep-IRAP.pdf` | `afad/plans/irap` |
| Gümüşhane İl Afet Risk Azaltma Planı | `Gumushane-IRAP-.pdf` | `afad/plans/irap` |
| Hakkari İl Afet Risk Azaltma Planı | `Hakkari-IRAP-GUNCEL-.pdf` | `afad/plans/irap` |
| Hatay İl Afet Risk Azaltma Planı | `HATAY-İRAP-2022.pdf` | `afad/plans/irap` |
| Isparta İl Afet Risk Azaltma Planı | `isparta_irap_.pdf` | `afad/plans/irap` |
| İstanbul İl Afet Risk Azaltma Planı | `irap_istanbul.pdf` | `afad/plans/irap` |
| İzmir İl Afet Risk Azaltma Planı | `IIRAP_Izmir-2025-_1_-a.pdf` | `afad/plans/irap` |
| Kahramanmaraş İl Afet Risk Azaltma Planı | `KAHRAMANMARAŞ-IL-AFET-RISK-AZALTMA-PLANI2022.pdf` | `afad/plans/irap` |
| Karabük İl Afet Risk Azaltma Planı (Eylemler) | `YALNIZ-EYLEMLER.pdf` | `afad/plans/irap` |
| Karaman İl Afet Risk Azaltma Planı | `IRAP-11102021-basim-yapilan.pdf` | `afad/plans/irap` |
| Kars İl Afet Risk Azaltma Planı | `KARS-IRAP.pdf` | `afad/plans/irap` |
| Kastamonu İl Afet Risk Azaltma Planı | `IRAP-KASTAMONU-2021.pdf` | `afad/plans/irap` |
| Kayseri İl Afet Risk Azaltma Planı | `Kayseri_IRAP_Baski.pdf` | `afad/plans/irap` |
| Kilis İl Afet Risk Azaltma Planı | `Kilis_IRAP.pdf` | `afad/plans/irap` |
| Kocaeli İl Afet Risk Azaltma Planı | `Kocaeli-IRAP.pdf` | `afad/plans/irap` |
| Konya İl Afet Risk Azaltma Planı | `KONYA-Il-IRAP-Plani-.pdf` | `afad/plans/irap` |
| Kütahya İl Afet Risk Azaltma Planı | `IRAP_2022.pdf` | `afad/plans/irap` |
| Kırklareli İl Afet Risk Azaltma Planı | `KIRKLARELI-IRAP.pdf` | `afad/plans/irap` |
| Kırıkkale İl Afet Risk Azaltma Planı | `Kirikkale-IRAP-Raporu-07102021.pdf` | `afad/plans/irap` |
| Kırşehir İl Afet Risk Azaltma Planı | `Kirsehir-IRAP_-Il-Afet-Risk-Azaltma-Plani.pdf` | `afad/plans/irap` |
| Manisa İl Afet Risk Azaltma Planı | `MANISA-IRAP.pdf` | `afad/plans/irap` |
| Mardin İl Afet Risk Azaltma Planı | `irap_Mardin.pdf` | `afad/plans/irap` |
| Mersin İl Afet Risk Azaltma Planı | `IRAP-MERSIN.pdf` | `afad/plans/irap` |
| Muğla İl Afet Risk Azaltma Planı | `Mugla_IRAP_20102021_Baski.pdf` | `afad/plans/irap` |
| Muş İl Afet Risk Azaltma Planı | `Mus-IRAP-Plani.pdf` | `afad/plans/irap` |
| Nevşehir İl Afet Risk Azaltma Planı | `NEVSEHIR-IRAP.pdf` | `afad/plans/irap` |
| Niğde İl Afet Risk Azaltma Planı | `NIGDE-IRAP.pdf` | `afad/plans/irap` |
| Ordu İl Afet Risk Azaltma Planı | `Ordu-Irap-Kurumlara-Dagitim.pdf` | `afad/plans/irap` |
| Rize İl Afet Risk Azaltma Planı | `Irap-.pdf` | `afad/plans/irap` |
| Samsun İl Afet Risk Azaltma Planı | `SAMSUN.pdf` | `afad/plans/irap` |
| Siirt İl Afet Risk Azaltma Planı | `SIIRT-IRAP.pdf` | `afad/plans/irap` |
| Sinop İl Afet Risk Azaltma Planı | `Sinop-IRAP.pdf` | `afad/plans/irap` |
| Sivas İl Afet Risk Azaltma Planı | `SivasIRAP.pdf` | `afad/plans/irap` |
| Tokat İl Afet Risk Azaltma Planı | `IRAP_Tokat_2022.pdf` | `afad/plans/irap` |
| Trabzon İl Afet Risk Azaltma Planı | `Trabzon-IRAP-03042026.pdf` | `afad/plans/irap` |
| Uşak İl Afet Risk Azaltma Planı | `USAK_IRAP_SON.pdf` | `afad/plans/irap` |
| Van İl Afet Risk Azaltma Planı | `VAN-IRAP.pdf` | `afad/plans/irap` |
| Yalova İl Afet Risk Azaltma Planı | `YALOVA-IRAP-2022.pdf` | `afad/plans/irap` |
| Yozgat İl Afet Risk Azaltma Planı | `Yozgat-IRAP.pdf` | `afad/plans/irap` |
| Zonguldak İl Afet Risk Azaltma Planı | `ZONGULDAK_IRAP.pdf` | `afad/plans/irap` |
| Çanakkale İl Afet Risk Azaltma Planı | `Canakkale-IRAP.pdf` | `afad/plans/irap` |
| Çankırı İl Afet Risk Azaltma Planı | `CANKIRI_IRAP.pdf` | `afad/plans/irap` |
| Çorum İl Afet Risk Azaltma Planı | `Corum_İRAP_SON.pdf` | `afad/plans/irap` |
| Şanlıurfa İl Afet Risk Azaltma Planı | `SANLIURFA_IRAPpdf.pdf` | `afad/plans/irap` |
| Şırnak İl Afet Risk Azaltma Planı | `IRAP-_Il-Afet-Risk-Azaltma-Plani_.pdf` | `afad/plans/irap` |

### Ulusal planlar (4)

| Başlık | Dosya | Klasör |
|---|---|---|
| Türkiye Afet Müdahale Planı (TAMP) | `TAMP.pdf` | `afad/plans/tamp` |
| Türkiye Afet Risk Azaltma Planı (TARAP) | `28032022-TARAP-kitap_V6.pdf` | `afad/plans/tarap` |
| Türkiye Afet Sonrası İyileştirme Planı (TASİP) | `23052025-TASIP.pdf` | `afad/plans/tasip` |
| Ulusal Deprem Stratejisi ve Eylem Planı (UDSEP) | `udsep_1402013_kitap.pdf` | `afad/plans/udsep` |

### Eğitim ve teknik yayınlar (19)

| Başlık | Dosya | Klasör |
|---|---|---|
| AFAD-RED | `afad_red.pdf` | `afad/publications` |
| Afet Bilgileri Envanteri | `abep_kitap_matbaa_final_04122008_small.pdf` | `afad/books` |
| Belediyeler için Sismik Mikrobölgeleme | `sismik-mikrobolgeleme_genisletilmis_ozet_tr.pdf` | `afad/books` |
| Belediyeler için Sismik Mikrobölgeleme | `sismik_mikrobolgeleme_ornek_uygulamalar_tr.pdf` | `afad/books` |
| Belediyeler için Sismik Mikrobölgeleme El Kitabı | `sismik_mikrobolgeleme_el_kitabi_tr.pdf` | `afad/books` |
| Deprem Haritası | `deprem_haritasi.pdf` | `afad/publications` |
| Deprem Takibi | `deprem_takibi.pdf` | `afad/publications` |
| Enkazda Arama ve Kurtarma Temel Eğitimi (2025) | `ENKAZDA-ARAMA-VE-KURTARMA-TEMEL-EĞİTİMİ-2025-.pdf` | `afad/publications` |
| Fayın Merkezi | `fayin_merkezi.pdf` | `afad/publications` |
| Gözlem Ağları | `gozlem_aglari.pdf` | `afad/publications` |
| Müdahale Kapasitesi | `mudahale_kapasitesi.pdf` | `afad/publications` |
| Olay Yeri Yönetimi Temel Eğitimi | `OLAY-YERİ-YÖNETİMİ-TEMEL-EĞİTİMİ-220925.pdf` | `afad/publications` |
| Risk Haritaları | `risk_haritalari.pdf` | `afad/publications` |
| Sismik Mikrobölgeleme Bilimsel Son Durum Raporu | `sismik_mikrobolgeleme_bilimsel_son_durum_raporu_tr.pdf` | `afad/books` |
| UDAP | `udap.pdf` | `afad/publications` |
| Yapısal Riskler ve Deprem Güvenliği | `yapisalriskler.pdf` | `afad/books` |
| Yerbilimsel Verilerin Planlamaya Entegrasyonu | `yerbilimsel_verilerin_planlamaya_entegrasyonu_tr.pdf` | `afad/books` |
| Yönetmelik Broşürü | `yonetmelik_brosur.pdf` | `afad/publications` |
| Yüzey Faylanması Tehlikesinin Değerlendirilmesi | `yft_min1.pdf` | `afad/books` |

### Raporlar (8)

| Başlık | Dosya | Klasör |
|---|---|---|
| 2020 Yılı Doğa Kaynaklı Olay İstatistikleri | `2020yilidogakaynakliolayistatistikleri.pdf` | `afad/statistics` |
| 2022 Yılı Doğa Kaynaklı Olay İstatistikleri | `2022-Yili-Doga-Kaynakli-Olay-Istatistikleri.pdf` | `afad/statistics` |
| 2023 Yılı Doğa Kaynaklı Olay İstatistikleri | `2023yilidogakaynakliolayistatistikleri-_1_.pdf` | `afad/statistics` |
| 2024 Yılı Doğa Kaynaklı Olay İstatistikleri | `2024-Yili-Doga-Kaynakli-Olay-Istatistikleri.pdf` | `afad/statistics` |
| Afet Sonrası İyileştirme Çalışmaları | `afet_sonrasi_iyilestirme_calismalari-1.pdf` | `afad/books` |
| Afete Dirençli Toplum Uygulama Örnekleri | `Afete_Direncli_Toplum_Uygulama_Ornekleri.pdf` | `afad/books` |
| İstanbul'da Afetler | `istanbulda-afetler.pdf` | `afad/books` |
| Müdahale Edilen Olaylar | `mudahale_edilen_olaylar.pdf` | `afad/publications` |

### Sözlükler (3)

| Başlık | Dosya | Klasör |
|---|---|---|
| Arama ve Kurtarma Açıklamalı Terimler Sözlüğü (2025) | `ARAMA-VE-KURTARMA-AÇIKLAMALI-TERİMLER-SÖZLÜĞÜ-2025.pdf` | `afad/publications` |
| Açıklamalı Afet Yönetimi Terimleri Sözlüğü | `Afet-Yönetim-Terimleri-Sözlüğü.pdf` | `afad/books` |
| Açıklamalı Afet Yönetimi Terimleri Sözlüğü | `sozluk.pdf` | `afad/books` |

### Bilgi görselleri (15)

| Başlık | Dosya | Klasör |
|---|---|---|
| 23 Ekim 2011 Van Depremi | `23 Ekim 2011 Van Depremi_.jpg` | `afad/infographics` |
| 70 2014050710924 presentation1 | `70-2014050710924-presentation1.pdf` | `afad/maps` |
| Afet Eğitim Yılı | `_afet_egitim_yili_5_milyon-_2_.jpeg` | `afad/infographics` |
| Afet Olay ve Zararlarının Dağılımı | `93-2014060215017-afetler_il_ilce_final_27102008_3.pdf` | `afad/maps` |
| Afet Olayları ve Zararları Dağılımı | `92-20140602121022-afetler_il_ilce_eng.pdf` | `afad/maps` |
| Afet ve Acil Durum Alanındaki Farkındalık, Bilinçlendirme ve Eğitim Çalışmaları | `Afet ve Acil Durum Alanındaki Farkındalık, Bilinçlendirme ve Eğitim Çalışmaları_.jpg` | `afad/infographics` |
| Afete Hazırlık için AFAD ve DASK El Ele | `Afete Hazırlık için AFAD DASK el ele____.jpg` | `afad/infographics` |
| Afetlerin Türkiye'deki Dağılımı | `94-201406021514-afetler_tum_dagilim_renkli_2.pdf` | `afad/maps` |
| Afetlerin Yıllara Göre Dağılımı | `95-2014060215230-afetler_yillar.pdf` | `afad/maps` |
| Avrupa Doğal Afetler Eğitim Merkezi (AFEM) Çalışmaları | `Avrupa Doğal Afetler Eğitim Merkezi (AFEM) Çalışmaları_.jpg` | `afad/infographics` |
| Bilgi Sistemleri ve Haberleşme Teknolojileri Kilometre Taşları | `Bilgi Sistemleri ve Haberleşme Teknolojileri Kilometre Taşları_.jpg` | `afad/infographics` |
| Kahramanmaraş Depremi Harcama Detayları | `Kahramanmaraş Depremi Bağış Kampanyası Harcama Detayları_.pdf` | `afad/infographics` |
| Kahramanmaraş Depremi Harcamaları | `Kahramanmaraş Depremi Bağış Kampanyası Harcamaları_.pdf` | `afad/infographics` |
| Tatbikatlar | `Tatbikatlar_.jpg` | `afad/infographics` |
| Türkiye Deprem Tehlike Haritası | `Türkiye Deprem Tehlike Haritası_.png` | `afad/infographics` |
