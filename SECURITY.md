# Güvenlik / Security

**Türkçe:** DARG'da bir güvenlik açığı bulduysanız (ör. kimlik doğrulamayı aşma, başkasının verisine erişme, istek
sınırlarını atlatıp maliyet doğurma) lütfen **herkese açık Issue açmayın**. GitHub'daki **Security → Report a
vulnerability** düğmesiyle özel olarak bildirin. Bildirimi aldığımızı birkaç gün içinde yazarız; düzeltme yayımlanana
kadar ayrıntıları paylaşmamanızı rica ederiz.

Tehlikeli ya da yanlış bir **cevap** (ör. yanlış ilk yardım tavsiyesi) bir güvenlik açığı değildir ama önemlidir:
başlığa **[güvenlik]** yazarak normal bir Issue açın (bkz. [CONTRIBUTING.md](CONTRIBUTING.md)).

**English:** please report vulnerabilities privately through **Security → Report a vulnerability** on GitHub, not in a
public issue. We will acknowledge within a few days. Dangerous or wrong *answers* are not vulnerabilities: open a
regular issue with **[güvenlik]** in the title.

Kapsam / scope: `src/api/` (accounts, announcements, limits), the deployment files in `deploy/` and the mobile app in
`mobile/`. The language models themselves are run by OpenAI and are out of scope.
