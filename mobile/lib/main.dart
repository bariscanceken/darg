import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'community.dart';
import 'emergency_card.dart';
import 'library.dart';
import 'screens/card_screen.dart';
import 'screens/chat_screen.dart';
import 'storage.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final storage = await Storage.open();
  final prefs = await SharedPreferences.getInstance();
  LibraryStore.instance = await LibraryStore.open(prefs, () => storage.serverUrl);
  CommunitySession.instance = CommunitySession(prefs, () => storage.serverUrl);
  CommunitySession.instance.restore(); // a saved organisation login, checked in the background
  final cards = EmergencyCardStore();
  cards.refreshLockScreen(); // Android may drop the notification after a restart
  runApp(DargApp(storage: storage, cards: cards));
}

class DargApp extends StatelessWidget {
  const DargApp({super.key, required this.storage, required this.cards});
  final Storage storage;
  final EmergencyCardStore cards;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<ThemeMode>(
      valueListenable: storage.themeMode,
      builder: (context, mode, _) => MaterialApp(
        title: 'DARG',
        debugShowCheckedModeBanner: false,
        theme: buildTheme(Brightness.light),
        darkTheme: buildTheme(Brightness.dark),
        themeMode: mode,
        home: storage.welcomed
            ? ChatScreen(storage: storage, cards: cards)
            : CardScreen(storage: storage, cards: cards, firstLaunch: true),
      ),
    );
  }
}
