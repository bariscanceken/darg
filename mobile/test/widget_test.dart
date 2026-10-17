import 'package:darg/emergency_card.dart';
import 'package:darg/main.dart';
import 'package:darg/models.dart';
import 'package:darg/screens/signal_screen.dart';
import 'package:darg/storage.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

Future<(Storage, EmergencyCardStore)> openStores({bool welcomed = true}) async {
  SharedPreferences.setMockInitialValues({'welcomed': welcomed});
  FlutterSecureStorage.setMockInitialValues({});
  return (await Storage.open(), EmergencyCardStore());
}

void main() {
  testWidgets('empty chat shows the greeting, suggestions and the 112 button', (tester) async {
    final (storage, cards) = await openStores();
    await tester.pumpWidget(DargApp(storage: storage, cards: cards));
    expect(find.text('Nasıl yardımcı olabilirim?'), findsOneWidget);
    expect(find.text('Deprem sırasında ne yapmalıyım?'), findsOneWidget);
    expect(find.text('112'), findsOneWidget);
  });

  testWidgets('first launch opens the emergency card, which can be skipped', (tester) async {
    final (storage, cards) = await openStores(welcomed: false);
    await tester.pumpWidget(DargApp(storage: storage, cards: cards));
    await tester.pumpAndSettle();
    expect(find.text('Acil durum kartınızı oluşturun'), findsOneWidget);
    await tester.scrollUntilVisible(
      find.text('Şimdilik geç'),
      200,
      scrollable: find.descendant(of: find.byType(ListView), matching: find.byType(Scrollable)).first,
    );
    await tester.tap(find.text('Şimdilik geç'));
    await tester.pumpAndSettle();
    expect(find.text('Nasıl yardımcı olabilirim?'), findsOneWidget);
    expect(storage.welcomed, isTrue);
  });

  testWidgets('the theme button switches dark and light and is remembered', (tester) async {
    final (storage, cards) = await openStores();
    await tester.pumpWidget(DargApp(storage: storage, cards: cards));
    expect(storage.themeMode.value, ThemeMode.dark);
    await tester.tap(find.byTooltip('Açık tema'));
    await tester.pumpAndSettle();
    expect(storage.themeMode.value, ThemeMode.light);
    expect(find.byTooltip('Koyu tema'), findsOneWidget);
    expect((await Storage.open()).themeMode.value, ThemeMode.light);
  });

  test('emergency card keeps its fields and summarises only the filled ones', () {
    const card = EmergencyCard(
      fullName: 'Ayşe Yılmaz',
      bloodType: '0 Rh+',
      contacts: [
        EmergencyContact(name: 'Mehmet', relation: 'eş', phone: '0555 000 00 00'),
        EmergencyContact(name: '', relation: '', phone: ''),
      ],
    );
    final copy = EmergencyCard.fromJson(card.toJson());
    expect(copy.bloodType, '0 Rh+');
    expect(copy.summaryLines, ['Ayşe Yılmaz', 'Kan grubu: 0 Rh+', 'Yakını: Mehmet (eş) 0555 000 00 00']);
    expect(const EmergencyCard().isEmpty, isTrue);
  });

  test('SOS pattern is three short, three long, three short', () {
    final flashes = [
      for (final (on, units) in sosPattern)
        if (on) units,
    ];
    expect(flashes, [1, 1, 1, 3, 3, 3, 1, 1, 1]);
  });

  test('API answer is parsed into sentences and numbered sources', () {
    final source = SourceRef.fromApi({
      'etiket': 'Kaynak 3',
      'tur': 'web',
      'kaynak': 'https://www.saglik.gov.tr/a',
      'sayfa': null,
    });
    expect(source.number, 3);
    expect(source.displayName, 'saglik.gov.tr');
    final sentence = Sentence.fromJson({'metin': 'Sakin olun.', 'kaynaklar': <int>[]});
    expect(sentence.sources, isEmpty);
  });

  test('conversations survive a save and load round trip', () async {
    final (storage, _) = await openStores();
    final conversation = Conversation(
      id: 'a',
      sessionId: 's',
      updatedAt: DateTime(2026, 10, 16),
      messages: [
        ChatMessage.user('deprem çantası'),
        ChatMessage.assistant(
          text: 'Su koyun. [Kaynak 1]',
          sentences: [
            Sentence(text: 'Su koyun.', sources: [1]),
          ],
          sources: [SourceRef(number: 1, kind: SourceKind.afad, source: 'ilk72saat1.pdf', page: 'Sayfa 4')],
          recordId: 'r1',
          rating: Rating.good,
        ),
      ],
    );
    await storage.saveConversations([conversation]);
    final loaded = storage.loadConversations().single;
    expect(loaded.title, 'deprem çantası');
    expect(loaded.messages.last.rating, Rating.good);
    expect(loaded.messages.last.sources.single.page, 'Sayfa 4');
  });
}
