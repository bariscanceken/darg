# DARG'ı sunucuya kurma (Ubuntu + Docker)

Bu rehber DARG'ın sunucu tarafını (cevap sistemi, kütüphane, duyurular) kiralık bir Ubuntu sunucuya kurar.
Kurulum bir kez yapılır; sonraki güncellemeler tek komuttur (`./deploy/guncelle.sh`).

## Gereksinimler

| | En az | Rahat |
|---|---|---|
| RAM (boşta) | 4 GB | 6 GB |
| Disk (boşta) | 15 GB | 20 GB |
| İşlemci | 2 çekirdek | 4 çekirdek |
| GPU | gerekmez | |
| İşletim sistemi | Ubuntu 22.04 / 24.04 | |

Büyük dil modeli OpenAI'de çalışır; sunucuda yalnız iki küçük arama modeli çalışır.

## 1. Sunucuya bağlan

Windows'ta PowerShell aç:

```
ssh kullanici@SUNUCU_IP
```

## 2. Docker'ı kur (bir kez)

```
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
exit
```

Tekrar `ssh` ile bağlan (grup değişikliği yeni oturumda geçerli olur) ve dene: `docker run --rm hello-world`

Sunucuda Docker zaten varsa (ör. arkadaşının sunucusu) bu adımı atla.

## 3. Kodu GitHub'dan çek (gizli depo için "deploy key")

Depo gizli olduğu için sunucuya yalnız bu depoyu **okuyabilen** bir anahtar tanımlanır:

```
ssh-keygen -t ed25519 -C darg-sunucu -f ~/.ssh/darg_deploy -N ""
cat ~/.ssh/darg_deploy.pub
```

Çıkan satırı kopyala → GitHub'da depo → **Settings → Deploy keys → Add deploy key** → yapıştır
("Allow write access" işaretleme). Sonra sunucuda:

```
cat >> ~/.ssh/config <<'EOF'
Host github-darg
  HostName github.com
  User git
  IdentityFile ~/.ssh/darg_deploy
EOF
git clone git@github-darg:bariscanceken/darg.git ~/darg
cd ~/darg
```

## 4. Verileri gönder (bir kez, ~2,5 GB)

Belgeler, index ve işlenmiş metinler git'te yok. **Windows'taki geliştirme bilgisayarında**, `C:\darg` klasöründe:

```
powershell -ExecutionPolicy Bypass -File deploy\veri_paketle.ps1
scp darg-veri.tar.gz kullanici@SUNUCU_IP:~/darg/
```

Sonra **sunucuda**:

```
cd ~/darg && tar -xzf darg-veri.tar.gz && rm darg-veri.tar.gz
```

## 5. Ayarlar (.env)

```
cp .env.example .env
nano .env
```

- `OPENAI_API_KEY=` OpenAI anahtarı
- `DOMAIN=` sunucu IP'sinin noktaları tire yapılmış hâli + `.sslip.io`.
  Örnek: IP `203.0.113.7` ise `DOMAIN=203-0-113-7.sslip.io`. Alan adı satın almaya gerek yok;
  ücretsiz HTTPS sertifikası otomatik alınır.

**80/443 portları boş mu?** (Arkadaşın sunucusunda başka bir site varsa dolu olabilir.)

```
sudo ss -tlnp | grep -E ':(80|443) '
```

- Çıktı **boşsa**: `.env` olduğu gibi kalsın (`COMPOSE_PROFILES=https`).
- Çıktı **doluysa**: `COMPOSE_PROFILES=` (boş) ve `API_BIND=0.0.0.0` yap. DARG `http://SUNUCU_IP:8000`
  adresinden çalışır (şifresiz http; test için yeterli). Kalıcı çözüm için sunucudaki mevcut web sunucusuna
  (nginx vb.) DARG'a yönlendiren bir kayıt eklenir.

## 6. Güvenlik duvarı

```
sudo ufw allow OpenSSH
sudo ufw allow 80,443/tcp      # https kullanılıyorsa
sudo ufw allow 8000/tcp        # yalnız API_BIND=0.0.0.0 ise
sudo ufw enable
```

## 7. Başlat

```
chmod +x deploy/guncelle.sh
./deploy/guncelle.sh
```

İlk seferde kutu (image) kurulur: 5–10 dakika. Sonunda `DARG çalışıyor` yazmalı. Dışarıdan dene:

```
curl https://DOMAIN/saglik        # {"durum":"ok"}
```

## 8. Yönetici hesabı (bir kez)

```
docker compose exec api python -m src.api.community create-admin --email senin@epostan --name "Ad Soyad"
```

Şifre sorulur; koda ya da dosyaya yazılmaz.

## 9. Telefondaki uygulama

Uygulamada **Menü → Hakkında ve ayarlar → Sunucu adresi**: `https://DOMAIN` → "Kaydet ve dene".

## 10. Web sitesi (darg.network)

Site `website/` klasöründe (düz HTML/CSS); Caddy sunar, ayrı bir kurulum gerekmez.

1. **DNS** (alan adını aldığın panelde), üç `A` kaydı, hepsi sunucu IP'si:
   `@` (darg.network), `www`, `api` → `SUNUCU_IP`
2. **.env**:
   ```
   DOMAIN=api.darg.network
   SITE_DOMAIN=darg.network, www.darg.network
   ```
   (`DOMAIN` değişince telefondaki sunucu adresi de `https://api.darg.network` olur.)
3. **APK** (git'te yok; bilgisayardan bir kez kopyalanır, her yeni sürümde tekrar):
   ```
   ssh root@SUNUCU_IP "mkdir -p /root/darg/data/indir"
   scp mobile/build/app/outputs/flutter-apk/app-release.apk root@SUNUCU_IP:/root/darg/data/indir/darg.apk
   ```
4. `./deploy/guncelle.sh` → dene: `https://darg.network` ve `https://darg.network/indir/darg.apk`

Sitede yapılan değişiklikler `git pull` ile gelir; Caddy dosyaları doğrudan klasörden okur.

---

## Günlük kullanım

| İş | Komut (sunucuda, `~/darg` içinde) |
|---|---|
| Güncelle (GitHub'a push'tan sonra) | `./deploy/guncelle.sh` |
| Kayıtları izle | `docker compose logs -f api` |
| Durum | `docker compose ps` |
| Durdur / başlat | `docker compose down` / `docker compose up -d` |
| Bellek ve disk | `free -h` / `df -h` |
| Veritabanı yedeği | `cp data/app/darg.db ~/darg-yedek-$(date +%F).db` |

Güncelleme verilere dokunmaz: kurum hesapları, duyurular ve kayıtlar `data/app` ve `data/logs` klasörlerinde kalır.
Belgeler ya da index yeniden üretilirse 4. adım tekrarlanır.

## Sorun giderme

- **`DARG 2 dakikada açılmadı`** → `docker compose logs --tail 50 api`. Çoğunlukla eksik veri (4. adım) ya da
  yetersiz bellek (`free -h`).
- **Cevaplarda 401 hatası** → OpenAI anahtarının süresi dolmuş; `.env` içinde yenile, `docker compose up -d`.
- **HTTPS sertifikası alınamıyor** → `DOMAIN` doğru mu, 80/443 portları açık mı: `docker compose logs caddy`.
