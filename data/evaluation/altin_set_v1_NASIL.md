# Altın test seti nasıl doldurulur

Bu set, sistemin doğruluğunu **senin** ölçütünle ölçmek için. Formdaki "cevap doğruluğu ≥ %85" hedefini
ancak bununla güvenilir biçimde gösterebiliriz.

**Dosya:** `altin_set_v1.csv` (Excel ile açılır; sütunlar noktalı virgülle ayrılır).

## Sütunlar
| Sütun | Ne yazılacak |
|---|---|
| `soru` | Bir vatandaşın gerçekten soracağı gibi, istersen yazım hatalı/günlük dille |
| `dogru_cevap_maddeleri` | Doğru cevabın 2–3 ana maddesi, aralarına `\|` koy. Kaynağa bakmadan, kendi bilgine göre doğru olanı yaz |
| `durum` | hazırlık / sırasında / sonrasında / enkaz altında / genel / geçmiş |
| `yer` | Soruda geçen il/ilçe (yoksa boş) |
| `cevaplanabilir_mi` | evet / hayır (hayır: sistem reddetmeli, ör. kişisel garanti, tahmin, konu dışı) |
| `not` | İstersen açıklama |

## Kurallar
- **30–50 soru** yeter. Karışık olsun: davranış (ev, dışarı, araç, sahil, enkaz), il soruları, tanımlar,
  geçmiş depremler, ilk yardım/psikolojik destek ve 5–8 tane "hayır" sorusu.
- Soruları **Claude'a göstermeden önce** yaz; set ayar yapmak için kullanılmaz, yalnızca ölçüm için kullanılır.
- Bu set **Final Test v2'den ayrıdır**; aynı soruları iki yere yazma.
- `ORNEK-` satırlarını silebilirsin.

## Puanlama (ölçümde)
Her cevap için sen ya da hakem şuna bakar: doğru maddeleri içeriyor mu, yanlış/tehlikeli bir şey söylüyor mu,
"hayır" sorusunu reddetti mi. Hakem önce senin 10–15 kararınla karşılaştırılıp kalibre edilir.
