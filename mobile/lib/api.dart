import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import 'models.dart';

/// Server address when none is saved in the settings. Override at build time:
///   flutter run --dart-define=DARG_API=http://192.168.1.20:8000
String defaultServerUrl() {
  const fromBuild = String.fromEnvironment('DARG_API');
  if (fromBuild.isNotEmpty) return fromBuild;
  if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
    return 'http://10.0.2.2:8000'; // the development computer, seen from the Android emulator
  }
  return 'http://localhost:8000';
}

class ApiException implements Exception {
  ApiException(this.message);
  final String message;
  @override
  String toString() => message;
}

class AnswerResult {
  AnswerResult({required this.message, required this.sessionId});
  final ChatMessage message;
  final String sessionId;
}

class DargApi {
  DargApi(this.baseUrl);

  final String baseUrl;
  static const _answerTimeout = Duration(seconds: 90); // web search answers can take ~30 s

  Uri _uri(String path) => Uri.parse('${baseUrl.replaceAll(RegExp(r'/+$'), '')}$path');

  Future<AnswerResult> ask(String question, {String? sessionId}) async {
    final http.Response response;
    try {
      response = await http
          .post(
            _uri('/soru'),
            headers: {'Content-Type': 'application/json; charset=utf-8'},
            body: jsonEncode({'soru': question, 'oturum_id': ?sessionId}),
          )
          .timeout(_answerTimeout);
    } on TimeoutException {
      throw ApiException('Cevap çok uzun sürdü. Lütfen tekrar deneyin.');
    } catch (_) {
      throw ApiException('Sunucuya ulaşılamadı. İnternet bağlantınızı ve sunucu adresini kontrol edin.');
    }
    if (response.statusCode != 200) {
      // The server explains known failures (e.g. 503: the AI service is unreachable) in "detail".
      String? detail;
      try {
        final body = jsonDecode(utf8.decode(response.bodyBytes));
        if (body is Map && body['detail'] is String) detail = body['detail'] as String;
      } catch (_) {}
      throw ApiException(detail ?? 'Sunucu bir hata döndürdü (${response.statusCode}). Lütfen tekrar deneyin.');
    }
    final json = jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    return AnswerResult(
      sessionId: json['oturum_id'] as String,
      message: ChatMessage.assistant(
        text: json['cevap'] as String,
        sentences: (json['cumleler'] as List? ?? []).map((s) => Sentence.fromJson(s)).toList(),
        sources: (json['kaynaklar'] as List).map((s) => SourceRef.fromApi(s)).toList(),
        recordId: json['kayit_id'] as String,
        usedWeb: json['web_kullanildi'] as bool? ?? false,
      ),
    );
  }

  Future<void> feedback(String recordId, Rating rating, {String? note}) async {
    final response = await http
        .post(
          _uri('/geri-bildirim'),
          headers: {'Content-Type': 'application/json; charset=utf-8'},
          body: jsonEncode({
            'kayit_id': recordId,
            'puan': rating == Rating.good ? 'iyi' : 'kotu',
            if (note != null && note.trim().isNotEmpty) 'not': note.trim(),
          }),
        )
        .timeout(const Duration(seconds: 15));
    if (response.statusCode != 200) {
      throw ApiException('Geri bildirim gönderilemedi.');
    }
  }

  Future<bool> health() async {
    try {
      final response = await http.get(_uri('/saglik')).timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
