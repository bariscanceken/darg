import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:url_launcher/url_launcher.dart';

import '../models.dart';
import '../screens/document_viewer_screen.dart';
import '../theme.dart';

TextStyle _answerStyle(DargPalette c) => TextStyle(fontSize: 16, height: 1.55, color: c.text);

class UserBubble extends StatelessWidget {
  const UserBubble({super.key, required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Align(
      alignment: Alignment.centerRight,
      child: ConstrainedBox(
        constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).width * 0.8),
        child: Container(
          margin: EdgeInsets.only(left: 40, top: 8, bottom: 8),
          padding: EdgeInsets.symmetric(horizontal: 16, vertical: 11),
          decoration: BoxDecoration(color: c.surfaceHigh, borderRadius: BorderRadius.circular(20)),
          child: SelectableText(text, style: TextStyle(fontSize: 16, height: 1.4, color: c.text)),
        ),
      ),
    );
  }
}

/// An answer the way Claude shows one: plain text on the background, sources and actions underneath.
class AssistantMessage extends StatelessWidget {
  const AssistantMessage({super.key, required this.message, required this.onRate, this.onRetry});

  final ChatMessage message;
  final void Function(Rating rating) onRate;
  final VoidCallback? onRetry;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    if (message.isError) return _ErrorNote(text: message.text, onRetry: onRetry);
    final cited = {for (final s in message.sentences) ...s.sources};
    final sources = message.sources.where((s) => cited.contains(s.number)).toList()
      ..sort((a, b) => a.number.compareTo(b.number));
    final hasGeneral = message.sentences.any((s) => s.sources.isEmpty);
    return Padding(
      padding: EdgeInsets.only(top: 8, bottom: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (message.sentences.isEmpty)
            SelectableText(message.text, style: _answerStyle(c))
          else
            for (final sentence in message.sentences)
              Padding(
                padding: EdgeInsets.only(bottom: 6),
                child: _SentenceText(sentence: sentence, sources: message.sources),
              ),
          if (sources.isNotEmpty) ...[SizedBox(height: 10), _SourceList(sources: sources)],
          if (hasGeneral || sources.any((s) => s.kind == SourceKind.web)) ...[
            SizedBox(height: 8),
            _Legend(general: hasGeneral, web: sources.any((s) => s.kind == SourceKind.web)),
          ],
          SizedBox(height: 4),
          _Actions(message: message, onRate: onRate),
        ],
      ),
    );
  }
}

class _SentenceText extends StatelessWidget {
  const _SentenceText({required this.sentence, required this.sources});
  final Sentence sentence;
  final List<SourceRef> sources;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Text.rich(
      TextSpan(
        style: _answerStyle(c),
        children: [
          TextSpan(text: sentence.text),
          if (sentence.sources.isEmpty)
            WidgetSpan(
              alignment: PlaceholderAlignment.middle,
              child: _Pill(label: 'genel bilgi', color: c.general),
            )
          else
            for (final number in sentence.sources)
              WidgetSpan(
                alignment: PlaceholderAlignment.middle,
                child: GestureDetector(
                  onTap: () {
                    final source = sources.where((s) => s.number == number).firstOrNull;
                    if (source != null) showSourceSheet(context, source);
                  },
                  child: _Pill(
                    label: '$number',
                    color: _kindColor(c, sources.where((s) => s.number == number).firstOrNull?.kind),
                  ),
                ),
              ),
        ],
      ),
    );
  }
}

Color _kindColor(DargPalette c, SourceKind? kind) => switch (kind) {
  SourceKind.web => c.web,
  SourceKind.card => c.accent,
  _ => c.afad,
};

class _Pill extends StatelessWidget {
  const _Pill({required this.label, required this.color});
  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.only(left: 5),
      padding: EdgeInsets.symmetric(horizontal: 6, vertical: 1),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color.withValues(alpha: 0.35)),
      ),
      child: Text(
        label,
        style: TextStyle(fontSize: 11.5, height: 1.3, color: color, fontWeight: FontWeight.w600),
      ),
    );
  }
}

class _SourceList extends StatelessWidget {
  const _SourceList({required this.sources});
  final List<SourceRef> sources;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        for (final source in sources)
          Padding(
            padding: EdgeInsets.only(bottom: 6),
            child: InkWell(
              borderRadius: BorderRadius.circular(10),
              onTap: () => showSourceSheet(context, source),
              child: Container(
                padding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                decoration: BoxDecoration(
                  color: c.surface,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: c.border),
                ),
                child: Row(
                  children: [
                    Text(
                      '${source.number}',
                      style: TextStyle(color: _kindColor(c, source.kind), fontWeight: FontWeight.w700, fontSize: 12.5),
                    ),
                    SizedBox(width: 8),
                    Icon(
                      source.kind == SourceKind.web ? Icons.public : Icons.description_outlined,
                      size: 15,
                      color: c.muted,
                    ),
                    SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        source.displayName,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(fontSize: 12.5, color: c.muted),
                      ),
                    ),
                    if (source.page != null) ...[
                      SizedBox(width: 8),
                      Text(
                        source.page!.replaceFirst('Sayfa ', 's. '),
                        style: TextStyle(fontSize: 12.5, color: c.muted),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ),
      ],
    );
  }
}

void showSourceSheet(BuildContext context, SourceRef source) {
  final c = context.colors;
  final kindLabel = switch (source.kind) {
    SourceKind.afad => 'AFAD belgesi',
    SourceKind.web => 'Resmî web sitesi',
    SourceKind.card => 'İl bilgi kartı (AFAD il planından derlendi)',
  };
  showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    builder: (context) => SafeArea(
      child: Padding(
        padding: EdgeInsets.fromLTRB(20, 0, 20, 20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Kaynak ${source.number} · $kindLabel',
              style: TextStyle(color: _kindColor(c, source.kind), fontWeight: FontWeight.w600),
            ),
            SizedBox(height: 10),
            Text(source.displayName, style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
            if (source.kind != SourceKind.web) ...[
              SizedBox(height: 6),
              Text(source.source, style: TextStyle(color: c.muted)),
            ],
            if (source.page != null) ...[SizedBox(height: 4), Text(source.page!, style: TextStyle(color: c.muted))],
            if (source.documentId != null) ...[
              const SizedBox(height: 14),
              FilledButton.icon(
                onPressed: () {
                  Navigator.pop(context);
                  DocumentViewerScreen.openById(context, source.documentId!, page: source.firstPage);
                },
                icon: const Icon(Icons.menu_book_outlined, size: 18),
                label: Text(source.firstPage != null ? 'Belgeyi aç (s. ${source.firstPage})' : 'Belgeyi aç'),
              ),
            ],
            if (source.kind == SourceKind.web) ...[
              SizedBox(height: 6),
              Text(source.source, style: TextStyle(color: c.web)),
              SizedBox(height: 14),
              FilledButton.icon(
                onPressed: () => launchUrl(Uri.parse(source.source), mode: LaunchMode.externalApplication),
                icon: Icon(Icons.open_in_new, size: 18),
                label: Text('Sayfayı aç'),
              ),
            ],
          ],
        ),
      ),
    ),
  );
}

class _Legend extends StatelessWidget {
  const _Legend({required this.general, required this.web});
  final bool general;
  final bool web;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final style = TextStyle(fontSize: 12, height: 1.4, color: c.muted);
    return Text.rich(
      TextSpan(
        style: style,
        children: [
          if (general) TextSpan(text: '"genel bilgi": AFAD belgelerinde doğrulanmamış cümle. '),
          if (web) TextSpan(text: 'Web bilgileri resmî sitelerden alındı; güncelliğini ilgili kurumdan kontrol edin.'),
        ],
      ),
    );
  }
}

class _Actions extends StatelessWidget {
  const _Actions({required this.message, required this.onRate});
  final ChatMessage message;
  final void Function(Rating rating) onRate;

  @override
  Widget build(BuildContext context) {
    final canRate = message.recordId != null;
    return Row(
      children: [
        _ActionIcon(
          icon: Icons.copy_rounded,
          tooltip: 'Kopyala',
          onTap: () {
            Clipboard.setData(ClipboardData(text: message.plainText));
            ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Cevap kopyalandı')));
          },
        ),
        if (canRate) ...[
          _ActionIcon(
            icon: message.rating == Rating.good ? Icons.thumb_up_alt : Icons.thumb_up_alt_outlined,
            tooltip: 'İyi cevap',
            active: message.rating == Rating.good,
            onTap: () => onRate(Rating.good),
          ),
          _ActionIcon(
            icon: message.rating == Rating.bad ? Icons.thumb_down_alt : Icons.thumb_down_alt_outlined,
            tooltip: 'Kötü cevap / hata bildir',
            active: message.rating == Rating.bad,
            onTap: () => onRate(Rating.bad),
          ),
        ],
      ],
    );
  }
}

class _ActionIcon extends StatelessWidget {
  const _ActionIcon({required this.icon, required this.tooltip, required this.onTap, this.active = false});
  final IconData icon;
  final String tooltip;
  final VoidCallback onTap;
  final bool active;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return IconButton(
      onPressed: onTap,
      tooltip: tooltip,
      visualDensity: VisualDensity.compact,
      iconSize: 18,
      color: active ? c.text : c.muted,
      icon: Icon(icon),
    );
  }
}

class _ErrorNote extends StatelessWidget {
  const _ErrorNote({required this.text, this.onRetry});
  final String text;
  final VoidCallback? onRetry;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Container(
      margin: EdgeInsets.symmetric(vertical: 8),
      padding: EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: c.emergency.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: c.emergency.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(text, style: TextStyle(color: c.text, height: 1.4)),
          SizedBox(height: 4),
          Text.rich(
            TextSpan(
              style: TextStyle(color: c.muted, fontSize: 13),
              children: [
                TextSpan(text: 'Acil bir durumdaysanız '),
                TextSpan(
                  text: '112\'yi arayın',
                  style: TextStyle(color: c.emergency, fontWeight: FontWeight.w600),
                  recognizer: TapGestureRecognizer()..onTap = () => launchUrl(Uri.parse('tel:112')),
                ),
                TextSpan(text: '.'),
              ],
            ),
          ),
          if (onRetry != null)
            Align(
              alignment: Alignment.centerRight,
              child: TextButton.icon(onPressed: onRetry, icon: Icon(Icons.refresh), label: Text('Tekrar dene')),
            ),
        ],
      ),
    );
  }
}
