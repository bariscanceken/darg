import 'package:darg/community.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('post from the API keeps organisation, place and validity', () {
    final post = Post.fromApi({
      'id': 7,
      'baslik': 'Su dağıtımı',
      'metin': 'Defne parkında 14:00.',
      'tur': 'yardim',
      'tur_adi': 'Yardım dağıtımı',
      'il': 'Hatay',
      'ilce': 'Defne',
      'gecerlilik': '2026-10-18T11:00:00+00:00',
      'durum': 'yayinda',
      'olusturma': '2026-10-17T08:00:00+00:00',
      'yayin': '2026-10-17T08:05:00+00:00',
      'inceleme_notu': null,
      'kurum_adi': 'Antakya Belediyesi',
      'kurum_guvenilir': true,
      'sikayet': 0,
    });
    expect(
      (post.place, post.orgName, post.kindName, post.orgTrusted),
      ('Defne, Hatay', 'Antakya Belediyesi', 'Yardım dağıtımı', true),
    );
    expect(post.validUntil.isUtc, isFalse); // shown in the phone's local time
  });

  test('account knows administrators', () {
    final account = Account.fromApi({
      'id': 1,
      'email': 'a@b.tr',
      'rol': 'yonetici',
      'durum': 'aktif',
      'kurum_adi': 'DARG Yönetim',
      'kurum_turu': 'diger',
      'yetkili_adi': 'Ad Soyad',
    });
    expect(account.isAdmin, isTrue);
  });

  test('time labels read naturally in Turkish', () {
    final now = DateTime(2026, 10, 17, 12, 0);
    expect(timeAgo(DateTime(2026, 10, 17, 11, 57), now: now), '3 dk önce');
    expect(timeAgo(DateTime(2026, 10, 17, 9, 0), now: now), '3 sa önce');
    expect(timeAgo(DateTime(2026, 10, 16, 9, 0), now: now), 'dün');
    expect(validLabel(DateTime(2026, 10, 17, 18, 0), now: now), 'Son geçerlilik: bugün 18:00');
    expect(validLabel(DateTime(2026, 10, 18, 9, 30), now: now), 'Son geçerlilik: yarın 09:30');
    expect(validLabel(DateTime(2026, 10, 20, 14, 0), now: now), 'Son geçerlilik: 20.10 14:00');
    expect(provinces, hasLength(81));
  });
}
