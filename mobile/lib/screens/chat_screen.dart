import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:uuid/uuid.dart';

import '../api.dart';
import '../community.dart';
import '../emergency_card.dart';
import '../models.dart';
import '../storage.dart';
import '../theme.dart';
import '../widgets/composer.dart';
import '../widgets/message_view.dart';
import '../widgets/thinking.dart';
import 'about_screen.dart';
import 'account_screen.dart';
import 'card_screen.dart';
import 'emergency_screen.dart';
import 'feed_screen.dart';
import 'library_screen.dart';
import 'signal_screen.dart';

const _suggestions = [
  'Deprem sırasında ne yapmalıyım?',
  'Deprem çantasında neler olmalı?',
  'Toplanma alanımı nasıl öğrenirim?',
  'Enkaz altında kalırsam ne yapmalıyım?',
];

Future<void> call112() => launchUrl(Uri.parse('tel:112'));

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key, required this.storage, required this.cards});
  final Storage storage;
  final EmergencyCardStore cards;

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _scroll = ScrollController();
  late List<Conversation> _conversations = widget.storage.loadConversations();
  late Conversation _current = _newConversation();
  bool _waiting = false;

  DargApi get _api => DargApi(widget.storage.serverUrl);

  Conversation _newConversation() => Conversation(id: Uuid().v4(), updatedAt: DateTime.now());

  void _persist() {
    _current.updatedAt = DateTime.now();
    _conversations = [_current, ..._conversations.where((c) => c.id != _current.id)];
    widget.storage.saveConversations(_conversations);
  }

  void _scrollToEnd() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _ask(String question) async {
    final conversation = _current;
    setState(() {
      conversation.messages.add(ChatMessage.user(question));
      _waiting = true;
    });
    _scrollToEnd();
    ChatMessage reply;
    try {
      final result = await _api.ask(question, sessionId: conversation.sessionId);
      conversation.sessionId = result.sessionId;
      reply = result.message;
    } on ApiException catch (error) {
      reply = ChatMessage.assistant(text: error.message, isError: true);
    }
    if (!mounted) return;
    setState(() {
      conversation.messages.add(reply);
      _waiting = false;
    });
    if (identical(conversation, _current)) {
      _persist();
      _scrollToEnd();
    }
  }

  void _retry() {
    final messages = _current.messages;
    if (messages.length < 2 || !messages.last.isError) return;
    final question = messages[messages.length - 2].text;
    setState(() => messages.removeRange(messages.length - 2, messages.length));
    _ask(question);
  }

  Future<void> _rate(ChatMessage message, Rating rating) async {
    String? note;
    if (rating == Rating.bad) {
      note = await _askNote();
      if (note == null) return; // cancelled
    }
    final previous = message.rating;
    setState(() => message.rating = rating);
    try {
      await _api.feedback(message.recordId!, rating, note: note);
      _persist();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Geri bildiriminiz için teşekkürler')));
      }
    } on Exception {
      if (!mounted) return;
      setState(() => message.rating = previous);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Geri bildirim gönderilemedi')));
    }
  }

  Future<String?> _askNote() {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Cevapta ne yanlıştı?'),
        content: TextField(
          controller: controller,
          autofocus: true,
          maxLines: 4,
          maxLength: 2000,
          decoration: InputDecoration(
            hintText: 'Yanlış, eksik ya da tehlikeli bulduğunuz kısmı yazın (isteğe bağlı)',
            border: OutlineInputBorder(),
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('Vazgeç')),
          FilledButton(onPressed: () => Navigator.pop(context, controller.text), child: Text('Gönder')),
        ],
      ),
    );
  }

  void _open(Conversation conversation) {
    setState(() {
      _current = conversation;
      _waiting = false;
    });
    Navigator.pop(context);
    _scrollToEnd();
  }

  void _startNew() {
    setState(() {
      _current = _newConversation();
      _waiting = false;
    });
  }

  void _delete(Conversation conversation) {
    setState(() {
      _conversations.removeWhere((c) => c.id == conversation.id);
      if (_current.id == conversation.id) _current = _newConversation();
    });
    widget.storage.saveConversations(_conversations);
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final messages = _current.messages;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Scaffold(
      appBar: AppBar(
        title: Text('DARG', style: TextStyle(fontWeight: FontWeight.w600, letterSpacing: 1.5)),
        actions: [
          IconButton(
            onPressed: widget.storage.toggleTheme,
            tooltip: isDark ? 'Açık tema' : 'Koyu tema',
            icon: Icon(isDark ? Icons.light_mode_outlined : Icons.dark_mode_outlined),
          ),
          IconButton(
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const SignalScreen())),
            tooltip: 'Yardım sinyali (düdük, SOS feneri)',
            icon: Icon(Icons.campaign_outlined, color: c.emergency),
          ),
          Padding(
            padding: EdgeInsets.only(right: 4),
            child: TextButton.icon(
              onPressed: call112,
              style: TextButton.styleFrom(
                foregroundColor: c.emergency,
                backgroundColor: c.emergency.withValues(alpha: 0.12),
                shape: StadiumBorder(),
                padding: EdgeInsets.symmetric(horizontal: 12),
              ),
              icon: Icon(Icons.phone_in_talk, size: 18),
              label: Text('112', style: TextStyle(fontWeight: FontWeight.w700)),
            ),
          ),
          IconButton(onPressed: _waiting ? null : _startNew, tooltip: 'Yeni sohbet', icon: Icon(Icons.edit_square)),
        ],
      ),
      drawer: _HistoryDrawer(
        conversations: _conversations,
        currentId: _current.id,
        onOpen: _open,
        onNew: () {
          Navigator.pop(context);
          _startNew();
        },
        onDelete: _delete,
        storage: widget.storage,
        cards: widget.cards,
      ),
      body: Column(
        children: [
          Expanded(
            child: messages.isEmpty
                ? _EmptyState(onPick: _ask)
                : ListView.builder(
                    controller: _scroll,
                    padding: EdgeInsets.fromLTRB(18, 8, 18, 16),
                    itemCount: messages.length + (_waiting ? 1 : 0),
                    itemBuilder: (context, index) {
                      if (index == messages.length) return ThinkingIndicator();
                      final message = messages[index];
                      if (message.fromUser) return UserBubble(text: message.text);
                      final isLast = index == messages.length - 1;
                      return AssistantMessage(
                        message: message,
                        onRate: (rating) => _rate(message, rating),
                        onRetry: isLast && !_waiting ? _retry : null,
                      );
                    },
                  ),
          ),
          Composer(onSend: _ask, enabled: !_waiting),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({required this.onPick});
  final void Function(String question) onPick;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return LayoutBuilder(
      builder: (context, constraints) => SingleChildScrollView(
        padding: EdgeInsets.symmetric(horizontal: 24),
        child: ConstrainedBox(
          constraints: BoxConstraints(minHeight: constraints.maxHeight),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.shield_outlined, size: 44, color: c.accent), // logo placeholder
              SizedBox(height: 18),
              Text(
                'Nasıl yardımcı olabilirim?',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 26, fontWeight: FontWeight.w500),
              ),
              SizedBox(height: 8),
              Text(
                'Deprem öncesi, sırası ve sonrası için AFAD kaynaklı bilgi',
                textAlign: TextAlign.center,
                style: TextStyle(color: c.muted, fontSize: 15),
              ),
              SizedBox(height: 28),
              Wrap(
                alignment: WrapAlignment.center,
                spacing: 8,
                runSpacing: 8,
                children: [
                  for (final suggestion in _suggestions)
                    ActionChip(
                      label: Text(suggestion),
                      onPressed: () => onPick(suggestion),
                      backgroundColor: c.surface,
                      side: BorderSide(color: c.border),
                      labelStyle: TextStyle(color: c.text, fontSize: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _HistoryDrawer extends StatelessWidget {
  const _HistoryDrawer({
    required this.conversations,
    required this.currentId,
    required this.onOpen,
    required this.onNew,
    required this.onDelete,
    required this.storage,
    required this.cards,
  });

  final List<Conversation> conversations;
  final String currentId;
  final void Function(Conversation) onOpen;
  final VoidCallback onNew;
  final void Function(Conversation) onDelete;
  final Storage storage;
  final EmergencyCardStore cards;

  void _go(BuildContext context, Widget screen) {
    Navigator.pop(context);
    Navigator.push(context, MaterialPageRoute(builder: (_) => screen));
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final saved = conversations.where((c) => c.messages.isNotEmpty).toList();
    return Drawer(
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: EdgeInsets.fromLTRB(12, 12, 12, 4),
              child: FilledButton.tonalIcon(
                onPressed: onNew,
                icon: Icon(Icons.add),
                label: Text('Yeni sohbet'),
                style: FilledButton.styleFrom(
                  backgroundColor: c.surfaceHigh,
                  foregroundColor: c.text,
                  padding: EdgeInsets.symmetric(vertical: 14),
                ),
              ),
            ),
            Padding(
              padding: EdgeInsets.fromLTRB(20, 16, 20, 6),
              child: Text('Geçmiş', style: TextStyle(color: c.muted, fontSize: 13)),
            ),
            Expanded(
              child: saved.isEmpty
                  ? Padding(
                      padding: EdgeInsets.symmetric(horizontal: 20),
                      child: Text('Henüz sohbet yok', style: TextStyle(color: c.muted)),
                    )
                  : ListView(
                      children: [
                        for (final conversation in saved)
                          ListTile(
                            dense: true,
                            selected: conversation.id == currentId,
                            selectedTileColor: c.surfaceHigh,
                            selectedColor: c.text,
                            title: Text(
                              conversation.title,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(fontSize: 15),
                            ),
                            onTap: () => onOpen(conversation),
                            trailing: IconButton(
                              icon: Icon(Icons.delete_outline, size: 20, color: c.muted),
                              tooltip: 'Sil',
                              onPressed: () => onDelete(conversation),
                            ),
                          ),
                      ],
                    ),
            ),
            Divider(height: 1),
            ListTile(
              leading: const Icon(Icons.campaign_outlined),
              title: const Text('Duyurular'),
              subtitle: Text('Doğrulanmış kurumlardan', style: TextStyle(color: c.muted, fontSize: 12)),
              onTap: () => _go(context, const FeedScreen()),
            ),
            ListTile(
              leading: const Icon(Icons.local_library_outlined),
              title: const Text('Kütüphane'),
              subtitle: Text('AFAD belgeleri, internetsiz okuma', style: TextStyle(color: c.muted, fontSize: 12)),
              onTap: () => _go(context, const LibraryScreen()),
            ),
            ListTile(
              leading: Icon(Icons.campaign_outlined, color: c.emergency),
              title: Text('Yardım sinyali'),
              subtitle: Text('Düdük, SOS feneri', style: TextStyle(color: c.muted, fontSize: 12)),
              onTap: () => _go(context, const SignalScreen()),
            ),
            ListTile(
              leading: Icon(Icons.badge_outlined),
              title: Text('Acil durum kartım'),
              subtitle: Text('Kan grubu, yakınlar', style: TextStyle(color: c.muted, fontSize: 12)),
              onTap: () => _go(context, CardScreen(storage: storage, cards: cards)),
            ),
            ListTile(
              leading: Icon(Icons.emergency_outlined, color: c.emergency),
              title: Text('Acil durum bilgileri'),
              subtitle: Text('İnternetsiz de açılır', style: TextStyle(color: c.muted, fontSize: 12)),
              onTap: () => _go(context, EmergencyScreen()),
            ),
            ListTile(
              leading: Icon(Icons.info_outline),
              title: Text('Hakkında ve ayarlar'),
              onTap: () => _go(context, AboutScreen(storage: storage)),
            ),
            ListenableBuilder(
              listenable: CommunitySession.instance,
              builder: (context, _) {
                final account = CommunitySession.instance.account;
                return ListTile(
                  leading: Icon(account == null ? Icons.login_rounded : Icons.verified_user_outlined),
                  title: Text(account == null ? 'Kurum girişi' : account.orgName),
                  subtitle: Text(
                    account == null ? 'Duyuru paylaşan kurumlar için' : (account.isAdmin ? 'Yönetici' : 'Kurum hesabı'),
                    style: TextStyle(color: c.muted, fontSize: 12),
                  ),
                  onTap: () => _go(context, const AccountScreen()),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}
