import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// One AFAD source document of the server's library (GET /kutuphane).
class LibraryDoc {
  const LibraryDoc({
    required this.id,
    required this.title,
    required this.category,
    required this.categoryName,
    required this.fileName,
    required this.isImage,
    required this.size,
    this.province,
    this.pages,
    this.searchable = false,
    this.core = false,
  });

  final String id;
  final String title;
  final String category;
  final String categoryName;
  final String? province;
  final String fileName;
  final bool isImage;
  final int size;
  final int? pages;
  final bool searchable;
  final bool core;

  String get sizeLabel => formatBytes(size);

  factory LibraryDoc.fromApi(Map<String, dynamic> json) => LibraryDoc(
    id: json['id'] as String,
    title: json['baslik'] as String,
    category: json['kategori'] as String,
    categoryName: json['kategori_adi'] as String,
    province: json['il'] as String?,
    fileName: json['dosya_adi'] as String,
    isImage: json['tur'] == 'gorsel',
    size: json['boyut'] as int,
    pages: json['sayfa'] as int?,
    searchable: json['aranabilir'] as bool? ?? false,
    core: json['temel'] as bool? ?? false,
  );

  Map<String, dynamic> toApi() => {
    'id': id,
    'baslik': title,
    'kategori': category,
    'kategori_adi': categoryName,
    'il': province,
    'dosya_adi': fileName,
    'tur': isImage ? 'gorsel' : 'pdf',
    'boyut': size,
    'sayfa': pages,
    'aranabilir': searchable,
    'temel': core,
  };
}

String formatBytes(int bytes) {
  final mb = bytes / (1024 * 1024);
  if (mb >= 10) return '${mb.round()} MB';
  if (mb >= 0.1) return '${mb.toStringAsFixed(1).replaceAll('.', ',')} MB';
  return '${(bytes / 1024).ceil()} KB';
}

class LibraryCatalog {
  const LibraryCatalog({required this.documents, required this.offline});
  final List<LibraryDoc> documents;
  final bool offline; // true: the server could not be reached; this is the last saved list

  List<LibraryDoc> get core => documents.where((d) => d.core).toList();
}

class SearchHit {
  const SearchHit({required this.doc, required this.page, required this.snippet});
  final LibraryDoc doc;
  final int page;
  final String snippet;
}

/// Turkish-aware folding for keyword search: "Gaz VANASI" and "gaz vanası" match, as do "İ"/"i" and "ı"/"I".
String foldTurkish(String text) {
  const from = 'İIıÇçĞğÖöŞşÜüÂâÎîÛû';
  const to = 'iiiccggoossuuaaiiuu';
  final buffer = StringBuffer();
  for (final rune in text.runes) {
    final char = String.fromCharCode(rune);
    final index = from.indexOf(char);
    buffer.write(index >= 0 ? to[index] : char.toLowerCase());
  }
  return buffer.toString();
}

/// Finds [query] in page texts; one hit per page with a short snippet around the first match.
List<SearchHit> searchPages(LibraryDoc doc, List<({int page, String text})> pages, String query, {int context = 70}) {
  // Words in order, anything non-alphanumeric between them: "çök kapan" finds "Çök, Kapan".
  final words = foldTurkish(query).split(RegExp(r'[^a-z0-9]+')).where((w) => w.isNotEmpty).toList();
  if (words.join().length < 2) return [];
  final pattern = RegExp(words.map(RegExp.escape).join(r'[^a-z0-9]+'));
  final hits = <SearchHit>[];
  for (final page in pages) {
    final flat = page.text.replaceAll(RegExp(r'\s+'), ' ');
    final match = pattern.firstMatch(foldTurkish(flat)); // folding keeps lengths, so positions line up
    if (match == null) continue;
    final start = (match.start - context).clamp(0, flat.length);
    final end = (match.end + context).clamp(0, flat.length);
    final snippet = '${start > 0 ? '…' : ''}${flat.substring(start, end).trim()}${end < flat.length ? '…' : ''}';
    hits.add(SearchHit(doc: doc, page: page.page, snippet: snippet));
  }
  return hits;
}

/// Library catalog, downloads and offline search. Downloads live in the app's documents folder:
/// `library/<id>.pdf|.jpg` and `library/<id>.text.json` (page texts for search). Not available on the web.
class LibraryStore extends ChangeNotifier {
  LibraryStore(this._prefs, this._serverUrl, this._dir);

  static late LibraryStore instance;
  static const _catalogKey = 'library_catalog_v1';
  static const _downloadsKey = 'library_downloads_v1';
  static const largeFileBytes = 50 * 1024 * 1024;

  final SharedPreferences _prefs;
  final String Function() _serverUrl;
  final Directory? _dir; // null on the web
  final Map<String, double> progress = {}; // id -> 0..1 while downloading
  final Map<String, List<({int page, String text})>> _texts = {};

  static Future<LibraryStore> open(SharedPreferences prefs, String Function() serverUrl) async {
    Directory? dir;
    if (!kIsWeb) {
      dir = Directory('${(await getApplicationDocumentsDirectory()).path}/library');
      await dir.create(recursive: true);
    }
    return LibraryStore(prefs, serverUrl, dir);
  }

  bool get canDownload => _dir != null;

  Uri _uri(String path) => Uri.parse('${_serverUrl().replaceAll(RegExp(r'/+$'), '')}$path');
  Uri fileUri(String id) => _uri('/kutuphane/$id/dosya');

  /// Downloaded documents (id -> catalog entry), kept so they can be listed without internet.
  Map<String, LibraryDoc> get downloads {
    final raw = _prefs.getString(_downloadsKey);
    if (raw == null) return {};
    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      return {for (final e in map.entries) e.key: LibraryDoc.fromApi(e.value as Map<String, dynamic>)};
    } catch (_) {
      return {};
    }
  }

  Future<void> _saveDownloads(Map<String, LibraryDoc> docs) =>
      _prefs.setString(_downloadsKey, jsonEncode({for (final e in docs.entries) e.key: e.value.toApi()}));

  bool isDownloaded(String id) => downloads.containsKey(id);
  int get usedBytes => downloads.values.fold(0, (sum, d) => sum + d.size);

  File? localFile(LibraryDoc doc) {
    if (_dir == null) return null;
    final dot = doc.fileName.lastIndexOf('.');
    final ext = dot >= 0 ? doc.fileName.substring(dot).toLowerCase() : '.pdf';
    return File('${_dir.path}/${doc.id}$ext');
  }

  File? _textFile(String id) => _dir == null ? null : File('${_dir.path}/$id.text.json');

  /// The server's list; without internet the last saved list (or only the downloads) with offline = true.
  Future<LibraryCatalog> loadCatalog() async {
    try {
      final response = await http.get(_uri('/kutuphane')).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        final body = utf8.decode(response.bodyBytes);
        await _prefs.setString(_catalogKey, body);
        return LibraryCatalog(documents: _parseCatalog(body), offline: false);
      }
    } catch (_) {}
    final cached = _prefs.getString(_catalogKey);
    return LibraryCatalog(documents: cached != null ? _parseCatalog(cached) : downloads.values.toList(), offline: true);
  }

  List<LibraryDoc> _parseCatalog(String body) =>
      ((jsonDecode(body) as Map<String, dynamic>)['belgeler'] as List).map((d) => LibraryDoc.fromApi(d)).toList();

  /// Streams the file to disk (with progress), then the page texts; throws on failure and leaves nothing behind.
  Future<void> download(LibraryDoc doc) async {
    final file = localFile(doc);
    if (file == null || progress.containsKey(doc.id)) return;
    final part = File('${file.path}.part');
    progress[doc.id] = 0;
    notifyListeners();
    final client = http.Client();
    try {
      final response = await client.send(http.Request('GET', fileUri(doc.id))).timeout(const Duration(seconds: 30));
      if (response.statusCode != 200) throw HttpException('HTTP ${response.statusCode}');
      final total = response.contentLength ?? doc.size;
      var received = 0;
      final sink = part.openWrite();
      try {
        await for (final chunk in response.stream) {
          sink.add(chunk);
          received += chunk.length;
          progress[doc.id] = total > 0 ? received / total : 0;
          notifyListeners();
        }
      } finally {
        await sink.close();
      }
      await part.rename(file.path);
      if (doc.searchable) {
        final text = await client.get(_uri('/kutuphane/${doc.id}/metin')).timeout(const Duration(seconds: 30));
        if (text.statusCode == 200) await _textFile(doc.id)!.writeAsBytes(text.bodyBytes);
      }
      await _saveDownloads({...downloads, doc.id: doc});
    } catch (_) {
      if (await part.exists()) await part.delete();
      rethrow;
    } finally {
      client.close();
      progress.remove(doc.id);
      notifyListeners();
    }
  }

  Future<void> remove(LibraryDoc doc) async {
    for (final file in [localFile(doc), _textFile(doc.id)]) {
      if (file != null && await file.exists()) await file.delete();
    }
    _texts.remove(doc.id);
    final docs = downloads..remove(doc.id);
    await _saveDownloads(docs);
    notifyListeners();
  }

  Future<List<({int page, String text})>> _pagesOf(String id) async {
    if (_texts.containsKey(id)) return _texts[id]!;
    final file = _textFile(id);
    var pages = <({int page, String text})>[];
    if (file != null && await file.exists()) {
      final json = jsonDecode(await file.readAsString()) as Map<String, dynamic>;
      pages = [for (final p in json['sayfalar'] as List) (page: p['sayfa'] as int, text: p['metin'] as String)];
    }
    return _texts[id] = pages;
  }

  /// Keyword search in the downloaded documents; works without internet.
  Future<List<SearchHit>> searchOffline(String query, {int limit = 100}) async {
    final hits = <SearchHit>[];
    for (final doc in downloads.values) {
      hits.addAll(searchPages(doc, await _pagesOf(doc.id), query));
      if (hits.length >= limit) break;
    }
    return hits.take(limit).toList();
  }
}
