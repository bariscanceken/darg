import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api.dart';
import 'models.dart';

/// Conversations and settings stay on the phone; the server only keeps the last turns of a session.
/// (The emergency card is kept apart, encrypted: see emergency_card.dart.)
class Storage {
  Storage(this._prefs);

  static const _conversationsKey = 'conversations_v1';
  static const _serverKey = 'server_url';
  static const _themeKey = 'theme';
  static const _welcomedKey = 'welcomed';
  static const maxConversations = 50;

  final SharedPreferences _prefs;

  static Future<Storage> open() async => Storage(await SharedPreferences.getInstance());

  /// Dark unless the user switched to light with the toggle in the top bar.
  late final ValueNotifier<ThemeMode> themeMode = ValueNotifier(
    _prefs.getString(_themeKey) == 'light' ? ThemeMode.light : ThemeMode.dark,
  );

  Future<void> toggleTheme() async {
    themeMode.value = themeMode.value == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    await _prefs.setString(_themeKey, themeMode.value == ThemeMode.dark ? 'dark' : 'light');
  }

  /// False until the first-launch emergency card screen was saved or skipped.
  bool get welcomed => _prefs.getBool(_welcomedKey) ?? false;

  Future<void> setWelcomed() => _prefs.setBool(_welcomedKey, true);

  List<Conversation> loadConversations() {
    final raw = _prefs.getString(_conversationsKey);
    if (raw == null) return [];
    try {
      final list = (jsonDecode(raw) as List).map((c) => Conversation.fromJson(c)).toList();
      list.sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
      return list;
    } catch (_) {
      return []; // an unreadable store must never block the app
    }
  }

  Future<void> saveConversations(List<Conversation> conversations) async {
    final kept = conversations.where((c) => c.messages.isNotEmpty).take(maxConversations).toList();
    await _prefs.setString(_conversationsKey, jsonEncode(kept.map((c) => c.toJson()).toList()));
  }

  String get serverUrl => _prefs.getString(_serverKey) ?? defaultServerUrl();

  Future<void> setServerUrl(String url) async {
    final trimmed = url.trim();
    if (trimmed.isEmpty) {
      await _prefs.remove(_serverKey);
    } else {
      await _prefs.setString(_serverKey, trimmed);
    }
  }
}
