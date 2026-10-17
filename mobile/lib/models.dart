/// Chat data kept on the phone and exchanged with the DARG API (src/api/app.py).
library;

enum SourceKind { afad, web, card }

class SourceRef {
  SourceRef({
    required this.number,
    required this.kind,
    required this.source,
    this.page,
    this.title,
    this.documentId,
    this.firstPage,
  });

  final int number;
  final SourceKind kind;
  final String source;
  final String? page;
  final String? title;
  final String? documentId; // library document (opens in the app's document viewer)
  final int? firstPage;

  /// "Kaynak 3" -> 3
  static int numberOf(String label) => int.tryParse(label.replaceAll(RegExp(r'[^0-9]'), '')) ?? 0;

  factory SourceRef.fromApi(Map<String, dynamic> json) => SourceRef(
    number: numberOf(json['etiket'] as String),
    kind: switch (json['tur']) {
      'web' => SourceKind.web,
      'il_karti' => SourceKind.card,
      _ => SourceKind.afad,
    },
    source: json['kaynak'] as String,
    page: json['sayfa'] as String?,
    title: json['baslik'] as String?,
    documentId: json['belge_id'] as String?,
    firstPage: json['ilk_sayfa'] as int?,
  );

  factory SourceRef.fromJson(Map<String, dynamic> json) => SourceRef(
    number: json['number'] as int,
    kind: SourceKind.values.byName(json['kind'] as String),
    source: json['source'] as String,
    page: json['page'] as String?,
    title: json['title'] as String?,
    documentId: json['documentId'] as String?,
    firstPage: json['firstPage'] as int?,
  );

  Map<String, dynamic> toJson() => {
    'number': number,
    'kind': kind.name,
    'source': source,
    'page': page,
    'title': title,
    'documentId': documentId,
    'firstPage': firstPage,
  };

  /// Short human label: document title, web host or province card.
  String get displayName {
    switch (kind) {
      case SourceKind.web:
        return Uri.tryParse(source)?.host.replaceFirst('www.', '') ?? source;
      case SourceKind.card:
        return 'İl bilgi kartı';
      case SourceKind.afad:
        return (title != null && title!.isNotEmpty) ? title! : source.replaceAll(RegExp(r'\.pdf$'), '');
    }
  }
}

class Sentence {
  Sentence({required this.text, required this.sources});

  final String text;
  final List<int> sources; // empty = general knowledge

  factory Sentence.fromJson(Map<String, dynamic> json) => Sentence(
    text: (json['metin'] ?? json['text']) as String,
    sources: ((json['kaynaklar'] ?? json['sources']) as List).cast<int>(),
  );

  Map<String, dynamic> toJson() => {'text': text, 'sources': sources};
}

enum Rating { none, good, bad }

class ChatMessage {
  ChatMessage.user(this.text)
    : fromUser = true,
      sentences = const [],
      sources = const [],
      recordId = null,
      usedWeb = false,
      rating = Rating.none,
      isError = false;

  ChatMessage.assistant({
    required this.text,
    this.sentences = const [],
    this.sources = const [],
    this.recordId,
    this.usedWeb = false,
    this.rating = Rating.none,
    this.isError = false,
  }) : fromUser = false;

  final bool fromUser;
  final String text;
  final List<Sentence> sentences;
  final List<SourceRef> sources;
  final String? recordId;
  final bool usedWeb;
  Rating rating;
  final bool isError;

  /// Text without the legend/notes the server appends (shown as UI elements instead).
  String get plainText => sentences.isNotEmpty ? sentences.map((s) => s.text).join('\n') : text;

  factory ChatMessage.fromJson(Map<String, dynamic> json) => json['fromUser'] == true
      ? ChatMessage.user(json['text'] as String)
      : ChatMessage.assistant(
          text: json['text'] as String,
          sentences: (json['sentences'] as List? ?? []).map((s) => Sentence.fromJson(s)).toList(),
          sources: (json['sources'] as List? ?? []).map((s) => SourceRef.fromJson(s)).toList(),
          recordId: json['recordId'] as String?,
          usedWeb: json['usedWeb'] as bool? ?? false,
          rating: Rating.values.byName(json['rating'] as String? ?? 'none'),
          isError: json['isError'] as bool? ?? false,
        );

  Map<String, dynamic> toJson() => {
    'fromUser': fromUser,
    'text': text,
    'sentences': sentences.map((s) => s.toJson()).toList(),
    'sources': sources.map((s) => s.toJson()).toList(),
    'recordId': recordId,
    'usedWeb': usedWeb,
    'rating': rating.name,
    'isError': isError,
  };
}

class Conversation {
  Conversation({required this.id, this.sessionId, required this.updatedAt, List<ChatMessage>? messages})
    : messages = messages ?? [];

  final String id;
  String? sessionId; // the server's oturum_id: follow-up questions keep their context
  DateTime updatedAt;
  final List<ChatMessage> messages;

  String get title {
    final first = messages.where((m) => m.fromUser).firstOrNull;
    return first == null ? 'Yeni sohbet' : first.text.trim().replaceAll('\n', ' ');
  }

  factory Conversation.fromJson(Map<String, dynamic> json) => Conversation(
    id: json['id'] as String,
    sessionId: json['sessionId'] as String?,
    updatedAt: DateTime.parse(json['updatedAt'] as String),
    messages: (json['messages'] as List).map((m) => ChatMessage.fromJson(m)).toList(),
  );

  Map<String, dynamic> toJson() => {
    'id': id,
    'sessionId': sessionId,
    'updatedAt': updatedAt.toIso8601String(),
    'messages': messages.map((m) => m.toJson()).toList(),
  };
}
