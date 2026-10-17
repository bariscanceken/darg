import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Personal emergency information. Blood type is health data (KVKK "özel nitelikli kişisel veri"):
/// the card is stored encrypted on the phone only and is never sent to the DARG server.
class EmergencyContact {
  const EmergencyContact({required this.name, required this.relation, required this.phone});

  final String name;
  final String relation;
  final String phone;

  bool get isEmpty => name.trim().isEmpty && phone.trim().isEmpty;

  factory EmergencyContact.fromJson(Map<String, dynamic> json) => EmergencyContact(
    name: json['name'] as String? ?? '',
    relation: json['relation'] as String? ?? '',
    phone: json['phone'] as String? ?? '',
  );

  Map<String, dynamic> toJson() => {'name': name, 'relation': relation, 'phone': phone};
}

class EmergencyCard {
  const EmergencyCard({
    this.fullName = '',
    this.bloodType = '',
    this.health = '',
    this.address = '',
    this.contacts = const [],
    this.showOnLockScreen = false,
  });

  final String fullName;
  final String bloodType;
  final String health; // chronic illnesses, allergies, medicines
  final String address;
  final List<EmergencyContact> contacts;
  final bool showOnLockScreen;

  static const bloodTypes = ['A Rh+', 'A Rh-', 'B Rh+', 'B Rh-', 'AB Rh+', 'AB Rh-', '0 Rh+', '0 Rh-'];

  bool get isEmpty =>
      fullName.trim().isEmpty &&
      bloodType.isEmpty &&
      health.trim().isEmpty &&
      address.trim().isEmpty &&
      contacts.every((c) => c.isEmpty);

  /// Lines shown on the card screen and the lock screen; empty fields are left out.
  List<String> get summaryLines => [
    if (fullName.trim().isNotEmpty) fullName.trim(),
    if (bloodType.isNotEmpty) 'Kan grubu: $bloodType',
    if (health.trim().isNotEmpty) 'Sağlık: ${health.trim()}',
    for (final contact in contacts.where((c) => !c.isEmpty))
      'Yakını: ${contact.name.trim()}${contact.relation.trim().isEmpty ? '' : ' (${contact.relation.trim()})'} '
          '${contact.phone.trim()}',
    if (address.trim().isNotEmpty) 'Adres: ${address.trim()}',
  ];

  factory EmergencyCard.fromJson(Map<String, dynamic> json) => EmergencyCard(
    fullName: json['fullName'] as String? ?? '',
    bloodType: json['bloodType'] as String? ?? '',
    health: json['health'] as String? ?? '',
    address: json['address'] as String? ?? '',
    contacts: (json['contacts'] as List? ?? []).map((c) => EmergencyContact.fromJson(c)).toList(),
    showOnLockScreen: json['showOnLockScreen'] as bool? ?? false,
  );

  Map<String, dynamic> toJson() => {
    'fullName': fullName,
    'bloodType': bloodType,
    'health': health,
    'address': address,
    'contacts': contacts.map((c) => c.toJson()).toList(),
    'showOnLockScreen': showOnLockScreen,
  };
}

class EmergencyCardStore {
  EmergencyCardStore({FlutterSecureStorage? secure, FlutterLocalNotificationsPlugin? notifications})
    : _secure = secure ?? const FlutterSecureStorage(),
      _notifications = notifications ?? FlutterLocalNotificationsPlugin();

  static const _key = 'emergency_card_v1';
  static const _notificationId = 112;

  final FlutterSecureStorage _secure;
  final FlutterLocalNotificationsPlugin _notifications;
  bool _initialized = false;

  Future<EmergencyCard> load() async {
    try {
      final raw = await _secure.read(key: _key);
      return raw == null ? const EmergencyCard() : EmergencyCard.fromJson(jsonDecode(raw));
    } catch (_) {
      return const EmergencyCard();
    }
  }

  /// Saves the card; returns false if the lock-screen note was wanted but could not be shown.
  Future<bool> save(EmergencyCard card) async {
    await _secure.write(key: _key, value: jsonEncode(card.toJson()));
    return _updateLockScreen(card);
  }

  Future<void> delete() async {
    await _secure.delete(key: _key);
    await _updateLockScreen(const EmergencyCard());
  }

  Future<void> refreshLockScreen() async {
    try {
      await _updateLockScreen(await load());
    } catch (_) {}
  }

  static bool get lockScreenSupported => !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

  /// Android: a silent, persistent notification that is visible on the lock screen, so someone who finds the
  /// phone (or a rescuer) can read the card without unlocking it.
  Future<bool> _updateLockScreen(EmergencyCard card) async {
    if (!lockScreenSupported) return !card.showOnLockScreen;
    if (!_initialized) {
      await _notifications.initialize(
        settings: const InitializationSettings(android: AndroidInitializationSettings('@mipmap/ic_launcher')),
      );
      _initialized = true;
    }
    if (!card.showOnLockScreen || card.isEmpty) {
      await _notifications.cancel(id: _notificationId);
      return true;
    }
    final android = _notifications.resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>();
    final allowed = await android?.requestNotificationsPermission() ?? true;
    if (!allowed) return false;
    final lines = card.summaryLines;
    await _notifications.show(
      id: _notificationId,
      title: 'ACİL DURUM KARTI',
      body: lines.take(3).join(' · '),
      notificationDetails: NotificationDetails(
        android: AndroidNotificationDetails(
          'emergency_card',
          'Acil durum kartı',
          channelDescription: 'Telefonu bulan kişinin kilit ekranında görebileceği acil durum bilgileri',
          importance: Importance.low,
          priority: Priority.low,
          ongoing: true,
          autoCancel: false,
          playSound: false,
          enableVibration: false,
          visibility: NotificationVisibility.public,
          category: AndroidNotificationCategory.status,
          styleInformation: BigTextStyleInformation(lines.join('\n')),
        ),
      ),
    );
    return true;
  }
}
