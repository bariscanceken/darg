import 'dart:convert';
import 'dart:io';

import 'package:darg/library.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _guide = {
  'id': 'abc123',
  'baslik': 'Afetlerde İlk 72 Saat Hazırlık Rehberi',
  'kategori': 'vatandas_rehberi',
  'kategori_adi': 'Vatandaş rehberleri',
  'il': null,
  'dosya_adi': 'ilk72saat1.pdf',
  'tur': 'pdf',
  'boyut': 4508876,
  'sayfa': 68,
  'aranabilir': true,
  'temel': true,
};

void main() {
  test('Turkish folding matches case and dotted/dotless i, keeping lengths', () {
    expect(foldTurkish('GAZ VANASI'), foldTurkish('gaz vanası'));
    expect(foldTurkish('İstanbul'), 'istanbul');
    expect(foldTurkish('Çök, Kapan, Tutun').length, 'Çök, Kapan, Tutun'.length);
  });

  test('page search returns one hit per page with a snippet', () {
    final doc = LibraryDoc.fromApi(_guide);
    final hits = searchPages(doc, [
      (page: 3, text: 'Deprem sonrası gaz kokusu alırsanız GAZ VANASINI kapatın ve pencereleri açın.'),
      (page: 4, text: 'Su ve yiyecek stoklayın.'),
    ], 'gaz vanası');
    expect(hits.single.page, 3);
    expect(hits.single.snippet, contains('VANASINI'));
    expect(searchPages(doc, [(page: 1, text: 'x')], 'g'), isEmpty); // too short
  });

  test('catalog entry keeps its fields through the download index', () {
    final doc = LibraryDoc.fromApi(_guide);
    final copy = LibraryDoc.fromApi(doc.toApi());
    expect((copy.title, copy.core, copy.pages, copy.isImage, copy.sizeLabel), (doc.title, true, 68, false, '4,3 MB'));
    expect(formatBytes(80 * 1024 * 1024), '80 MB');
  });

  test('downloaded documents are searched without internet', () async {
    final dir = await Directory.systemTemp.createTemp('darg_library_test');
    addTearDown(() => dir.delete(recursive: true));
    SharedPreferences.setMockInitialValues({
      'library_downloads_v1': jsonEncode({'abc123': _guide}),
    });
    final store = LibraryStore(await SharedPreferences.getInstance(), () => 'http://127.0.0.1:9', dir);
    File('${dir.path}/abc123.text.json').writeAsStringSync(
      jsonEncode({
        'sayfalar': [
          {'sayfa': 12, 'metin': 'Deprem anında Çök, Kapan, Tutun hareketini yapın.'},
        ],
      }),
    );
    expect(store.isDownloaded('abc123'), isTrue);
    expect(store.localFile(store.downloads['abc123']!)!.path, endsWith('abc123.pdf'));
    final hits = await store.searchOffline('çök kapan');
    expect((hits.single.doc.id, hits.single.page), ('abc123', 12));
    final catalog = await store.loadCatalog(); // server unreachable -> offline, downloads still listed
    expect((catalog.offline, catalog.documents.single.id), (true, 'abc123'));
    await store.remove(catalog.documents.single);
    expect(store.downloads, isEmpty);
  });
}
