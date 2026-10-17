import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

import 'api.dart';

/// Verified announcements (server: src/api/community.py). Citizens read and report without an account;
/// organisations apply, log in and post; administrators moderate.

const postKinds = {
  'duyuru': 'Duyuru',
  'toplanma': 'Toplanma alanı',
  'yardim': 'Yardım dağıtımı',
  'saglik': 'Sağlık',
  'uyari': 'Uyarı',
};

const orgTypes = {
  'afad': 'AFAD birimi',
  'belediye': 'Belediye',
  'kamu': 'Diğer kamu kurumu',
  'saglik': 'Sağlık kuruluşu',
  'stk': 'Sivil toplum kuruluşu',
  'diger': 'Diğer',
};

class Account {
  const Account({
    required this.id,
    required this.email,
    required this.role,
    required this.status,
    required this.orgName,
    required this.orgType,
    required this.contactName,
    this.phone = '',
    this.proof = '',
    this.trusted = false,
    this.createdAt,
    this.reviewNote,
  });

  final int id;
  final String email;
  final String role; // kurum | yonetici
  final String status; // beklemede | aktif | reddedildi | askida
  final String orgName;
  final String orgType;
  final String contactName;
  final String phone;
  final String proof;
  final bool trusted;
  final DateTime? createdAt;
  final String? reviewNote;

  bool get isAdmin => role == 'yonetici';

  factory Account.fromApi(Map<String, dynamic> json) => Account(
    id: json['id'] as int,
    email: json['email'] as String,
    role: json['rol'] as String,
    status: json['durum'] as String,
    orgName: json['kurum_adi'] as String,
    orgType: json['kurum_turu'] as String,
    contactName: json['yetkili_adi'] as String,
    phone: json['telefon'] as String? ?? '',
    proof: json['kanit'] as String? ?? '',
    trusted: json['guvenilir'] as bool? ?? false,
    createdAt: DateTime.tryParse(json['olusturma'] as String? ?? ''),
    reviewNote: json['inceleme_notu'] as String?,
  );
}

class Post {
  const Post({
    required this.id,
    required this.title,
    required this.body,
    required this.kind,
    required this.kindName,
    required this.validUntil,
    required this.status,
    required this.createdAt,
    this.province,
    this.district,
    this.publishedAt,
    this.reviewNote,
    this.orgName,
    this.orgTrusted = false,
    this.reports = 0,
    this.reportReasons = const [],
  });

  final int id;
  final String title;
  final String body;
  final String kind;
  final String kindName;
  final String? province;
  final String? district;
  final DateTime validUntil;
  final String status; // beklemede | yayinda | reddedildi | kaldirildi | gizlendi
  final DateTime createdAt;
  final DateTime? publishedAt;
  final String? reviewNote;
  final String? orgName;
  final bool orgTrusted;
  final int reports;
  final List<String> reportReasons;

  String get place => [district, province].whereType<String>().where((s) => s.isNotEmpty).join(', ');

  factory Post.fromApi(Map<String, dynamic> json) => Post(
    id: json['id'] as int,
    title: json['baslik'] as String,
    body: json['metin'] as String,
    kind: json['tur'] as String,
    kindName: json['tur_adi'] as String? ?? postKinds[json['tur']] ?? '',
    province: json['il'] as String?,
    district: json['ilce'] as String?,
    validUntil: DateTime.parse(json['gecerlilik'] as String).toLocal(),
    status: json['durum'] as String,
    createdAt: DateTime.parse(json['olusturma'] as String).toLocal(),
    publishedAt: json['yayin'] == null ? null : DateTime.parse(json['yayin'] as String).toLocal(),
    reviewNote: json['inceleme_notu'] as String?,
    orgName: json['kurum_adi'] as String?,
    orgTrusted: json['kurum_guvenilir'] as bool? ?? false,
    reports: json['sikayet'] as int? ?? 0,
    reportReasons: (json['sikayet_nedenleri'] as List? ?? []).cast<String>(),
  );
}

/// "3 dk önce", "2 sa önce", "dün", "12.10.2026"
String timeAgo(DateTime moment, {DateTime? now}) {
  final diff = (now ?? DateTime.now()).difference(moment);
  if (diff.inMinutes < 1) return 'az önce';
  if (diff.inMinutes < 60) return '${diff.inMinutes} dk önce';
  if (diff.inHours < 24) return '${diff.inHours} sa önce';
  if (diff.inDays == 1) return 'dün';
  if (diff.inDays < 7) return '${diff.inDays} gün önce';
  return '${moment.day.toString().padLeft(2, '0')}.${moment.month.toString().padLeft(2, '0')}.${moment.year}';
}

/// "Son geçerlilik: bugün 18:00", "… yarın 09:30", "… 20.10 14:00"
String validLabel(DateTime until, {DateTime? now}) {
  final today = now ?? DateTime.now();
  final hh = '${until.hour.toString().padLeft(2, '0')}:${until.minute.toString().padLeft(2, '0')}';
  final days = DateTime(until.year, until.month, until.day).difference(DateTime(today.year, today.month, today.day));
  final day = switch (days.inDays) {
    0 => 'bugün',
    1 => 'yarın',
    _ => '${until.day.toString().padLeft(2, '0')}.${until.month.toString().padLeft(2, '0')}',
  };
  return 'Son geçerlilik: $day $hh';
}

/// Logged-in organisation or administrator; the token lives in encrypted storage.
class CommunitySession extends ChangeNotifier {
  CommunitySession(this._prefs, this._serverUrl, {FlutterSecureStorage? secure})
    : _secure = secure ?? const FlutterSecureStorage();

  static late CommunitySession instance;
  static const _tokenKey = 'community_token';
  static const _deviceKey = 'device_id';

  final SharedPreferences _prefs;
  final String Function() _serverUrl;
  final FlutterSecureStorage _secure;
  String? _token;
  Account? account;

  bool get loggedIn => account != null;

  /// Anonymous id so a phone can report a post only once.
  String get deviceId {
    var id = _prefs.getString(_deviceKey);
    if (id == null) {
      id = const Uuid().v4();
      _prefs.setString(_deviceKey, id);
    }
    return id;
  }

  Uri _uri(String path) => Uri.parse('${_serverUrl().replaceAll(RegExp(r'/+$'), '')}$path');

  Map<String, String> get _headers => {
    'Content-Type': 'application/json; charset=utf-8',
    if (_token != null) 'Authorization': 'Bearer $_token',
  };

  Future<dynamic> _send(String method, String path, [Map<String, dynamic>? body]) async {
    final http.Response response;
    try {
      final request = http.Request(method, _uri(path))..headers.addAll(_headers);
      if (body != null) request.body = jsonEncode(body);
      response = await http.Response.fromStream(await request.send().timeout(const Duration(seconds: 20)));
    } catch (_) {
      throw ApiException('Sunucuya ulaşılamadı. İnternet bağlantınızı kontrol edin.');
    }
    final text = utf8.decode(response.bodyBytes);
    final json = text.isEmpty ? null : jsonDecode(text);
    if (response.statusCode >= 400) {
      if (response.statusCode == 401 && _token != null) await _forget();
      final detail = json is Map ? json['detail'] : null;
      throw ApiException(detail is String ? detail : 'İşlem yapılamadı (${response.statusCode}).');
    }
    return json;
  }

  Future<void> _forget() async {
    _token = null;
    account = null;
    await _secure.delete(key: _tokenKey).catchError((_) {});
    notifyListeners();
  }

  /// Restores a saved login (silently drops it if the server says it expired).
  Future<void> restore() async {
    try {
      _token = await _secure.read(key: _tokenKey);
      if (_token == null) return;
      account = Account.fromApi(await _send('GET', '/hesap/ben') as Map<String, dynamic>);
      notifyListeners();
    } catch (_) {}
  }

  Future<void> login(String email, String password) async {
    final json = await _send('POST', '/hesap/giris', {'email': email.trim(), 'sifre': password}) as Map;
    _token = json['token'] as String;
    account = Account.fromApi(json['hesap'] as Map<String, dynamic>);
    await _secure.write(key: _tokenKey, value: _token);
    notifyListeners();
  }

  Future<void> logout() async {
    try {
      await _send('POST', '/hesap/cikis');
    } catch (_) {}
    await _forget();
  }

  Future<String> apply(Map<String, dynamic> application) async =>
      ((await _send('POST', '/hesap/basvuru', application)) as Map)['mesaj'] as String;

  Future<List<Post>> feed({String? province}) async {
    final query = province == null ? '' : '?il=${Uri.encodeQueryComponent(province)}';
    return _posts(await _send('GET', '/akis$query'));
  }

  Future<String> createPost(Map<String, dynamic> post) async =>
      ((await _send('POST', '/gonderiler', post)) as Map)['mesaj'] as String;

  Future<List<Post>> myPosts() async => _posts(await _send('GET', '/gonderiler/benim'));

  Future<void> deletePost(int id) => _send('DELETE', '/gonderiler/$id');

  Future<String> report(int id, String reason) async =>
      ((await _send('POST', '/gonderiler/$id/sikayet', {'cihaz_id': deviceId, 'neden': reason})) as Map)['durum']
          as String;

  // Administration
  Future<Map<String, dynamic>> adminSummary() async => (await _send('GET', '/yonetim/ozet')) as Map<String, dynamic>;

  Future<List<Account>> adminOrganisations({String? status}) async {
    final json = await _send('GET', '/yonetim/kurumlar${status == null ? '' : '?durum=$status'}') as Map;
    return (json['kurumlar'] as List).map((a) => Account.fromApi(a)).toList();
  }

  Future<void> adminDecideOrganisation(int id, String decision, {bool trusted = false, String? note}) =>
      _send('POST', '/yonetim/kurumlar/$id', {'karar': decision, 'guvenilir': trusted, 'not': ?note});

  Future<void> adminTrust(int id, bool trusted) => _send('POST', '/yonetim/kurumlar/$id/guven', {'guvenilir': trusted});

  /// Every post with open reports (still published or hidden by reports), most reported first.
  Future<List<Post>> adminReported() async => _posts(await _send('GET', '/yonetim/sikayetler'));

  Future<List<Post>> adminPosts(String status) async => _posts(await _send('GET', '/yonetim/gonderiler?durum=$status'));

  Future<void> adminDecidePost(int id, String decision, {String? note}) =>
      _send('POST', '/yonetim/gonderiler/$id', {'karar': decision, 'not': ?note});

  List<Post> _posts(dynamic json) =>
      ((json as Map)['gonderiler'] as List).map((p) => Post.fromApi(p as Map<String, dynamic>)).toList();
}

/// The 81 provinces, for the feed filter and the post form.
const provinces = [
  'Adana', 'Adıyaman', 'Afyonkarahisar', 'Ağrı', 'Aksaray', 'Amasya', 'Ankara', 'Antalya', 'Ardahan', 'Artvin', //
  'Aydın', 'Balıkesir', 'Bartın', 'Batman', 'Bayburt', 'Bilecik', 'Bingöl', 'Bitlis', 'Bolu', 'Burdur', 'Bursa', //
  'Çanakkale', 'Çankırı', 'Çorum', 'Denizli', 'Diyarbakır', 'Düzce', 'Edirne', 'Elazığ', 'Erzincan', 'Erzurum', //
  'Eskişehir', 'Gaziantep', 'Giresun', 'Gümüşhane', 'Hakkari', 'Hatay', 'Iğdır', 'Isparta', 'İstanbul', 'İzmir', //
  'Kahramanmaraş', 'Karabük', 'Karaman', 'Kars', 'Kastamonu', 'Kayseri', 'Kilis', 'Kırıkkale', 'Kırklareli', //
  'Kırşehir', 'Kocaeli', 'Konya', 'Kütahya', 'Malatya', 'Manisa', 'Mardin', 'Mersin', 'Muğla', 'Muş', 'Nevşehir', //
  'Niğde', 'Ordu', 'Osmaniye', 'Rize', 'Sakarya', 'Samsun', 'Şanlıurfa', 'Siirt', 'Sinop', 'Şırnak', 'Sivas', //
  'Tekirdağ', 'Tokat', 'Trabzon', 'Tunceli', 'Uşak', 'Van', 'Yalova', 'Yozgat', 'Zonguldak',
];
