import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../api.dart';
import '../community.dart';
import '../theme.dart';
import 'account_screen.dart';

/// Announcements from verified organisations, newest first, optionally for one province.
class FeedScreen extends StatefulWidget {
  const FeedScreen({super.key});

  @override
  State<FeedScreen> createState() => _FeedScreenState();
}

class _FeedScreenState extends State<FeedScreen> {
  static const _provinceKey = 'feed_province';
  final _session = CommunitySession.instance;
  List<Post>? _posts;
  String? _error;
  String? _province;

  @override
  void initState() {
    super.initState();
    SharedPreferences.getInstance().then((prefs) {
      _province = prefs.getString(_provinceKey);
      _load();
    });
  }

  Future<void> _load() async {
    try {
      final posts = await _session.feed(province: _province);
      if (mounted) {
        setState(() {
          _posts = posts;
          _error = null;
        });
      }
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  Future<void> _pickProvince() async {
    final picked = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => const ProvincePicker(allowAll: true),
    );
    if (picked == null) return;
    final province = picked.isEmpty ? null : picked;
    final prefs = await SharedPreferences.getInstance();
    province == null ? await prefs.remove(_provinceKey) : await prefs.setString(_provinceKey, province);
    setState(() {
      _province = province;
      _posts = null;
    });
    _load();
  }

  Future<void> _report(Post post) async {
    final controller = TextEditingController();
    final reason = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Gönderiyi şikâyet et'),
        content: TextField(
          controller: controller,
          maxLines: 3,
          maxLength: 500,
          decoration: const InputDecoration(
            hintText: 'Ne yanlış? (ör. adres hatalı, süresi geçmiş, sahte)',
            border: OutlineInputBorder(),
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Vazgeç')),
          FilledButton(onPressed: () => Navigator.pop(context, controller.text), child: const Text('Gönder')),
        ],
      ),
    );
    if (reason == null || !mounted) return;
    try {
      await _session.report(post.id, reason);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Teşekkürler. Şikâyetiniz yöneticilere iletildi; karar yöneticinindir.')),
      );
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return ListenableBuilder(
      listenable: _session,
      builder: (context, _) {
        final account = _session.account;
        return Scaffold(
          appBar: AppBar(
            title: const Text('Duyurular'),
            actions: [
              Padding(
                padding: const EdgeInsets.only(right: 8),
                child: TextButton.icon(
                  icon: Icon(account == null ? Icons.login_rounded : Icons.verified_user_outlined, size: 18),
                  label: Text(account == null ? 'Kurum girişi' : 'Hesabım'),
                  onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const AccountScreen())),
                ),
              ),
            ],
          ),
          floatingActionButton: account == null
              ? null
              : FloatingActionButton.extended(
                  onPressed: () async {
                    await Navigator.push(context, MaterialPageRoute(builder: (_) => const ComposePostScreen()));
                    _load();
                  },
                  icon: const Icon(Icons.edit_outlined),
                  label: const Text('Duyuru yaz'),
                ),
          body: RefreshIndicator(
            onRefresh: _load,
            child: ListView(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 96),
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        'Doğrulanmış kurumların duyuruları. Her gönderi yönetici kontrolünden geçer.',
                        style: TextStyle(color: c.muted, fontSize: 13, height: 1.4),
                      ),
                    ),
                    const SizedBox(width: 8),
                    ActionChip(
                      avatar: const Icon(Icons.place_outlined, size: 18),
                      label: Text(_province ?? 'Tüm iller'),
                      onPressed: _pickProvince,
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                if (_error != null)
                  _Message(icon: Icons.wifi_off_rounded, text: _error!, action: ('Tekrar dene', _load))
                else if (_posts == null)
                  const Padding(
                    padding: EdgeInsets.all(40),
                    child: Center(child: CircularProgressIndicator()),
                  )
                else if (_posts!.isEmpty)
                  _Message(
                    icon: Icons.campaign_outlined,
                    text: _province == null
                        ? 'Şu anda yayında duyuru yok.'
                        : '$_province için şu anda yayında duyuru yok.',
                  )
                else
                  for (final post in _posts!) PostCard(post: post, onReport: () => _report(post)),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _Message extends StatelessWidget {
  const _Message({required this.icon, required this.text, this.action});
  final IconData icon;
  final String text;
  final (String, VoidCallback)? action;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 16),
      child: Column(
        children: [
          Icon(icon, size: 40, color: c.muted),
          const SizedBox(height: 12),
          Text(
            text,
            textAlign: TextAlign.center,
            style: TextStyle(color: c.muted, height: 1.4),
          ),
          if (action != null) TextButton(onPressed: action!.$2, child: Text(action!.$1)),
        ],
      ),
    );
  }
}

Color kindColor(DargPalette c, String kind) => switch (kind) {
  'uyari' => c.emergency,
  'saglik' => c.afad,
  'toplanma' => c.web,
  'yardim' => c.accent,
  _ => c.muted,
};

IconData kindIcon(String kind) => switch (kind) {
  'uyari' => Icons.warning_amber_rounded,
  'saglik' => Icons.local_hospital_outlined,
  'toplanma' => Icons.groups_outlined,
  'yardim' => Icons.volunteer_activism_outlined,
  _ => Icons.campaign_outlined,
};

/// One announcement. [footer] replaces the report button (e.g. moderation buttons or the post's status).
class PostCard extends StatelessWidget {
  const PostCard({super.key, required this.post, this.onReport, this.footer});
  final Post post;
  final VoidCallback? onReport;
  final Widget? footer;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final color = kindColor(c, post.kind);
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.fromLTRB(14, 12, 6, 6),
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: c.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(color: color.withValues(alpha: 0.14), borderRadius: BorderRadius.circular(6)),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(kindIcon(post.kind), size: 14, color: color),
                    const SizedBox(width: 4),
                    Text(
                      post.kindName,
                      style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600),
                    ),
                  ],
                ),
              ),
              if (post.place.isNotEmpty) ...[
                const SizedBox(width: 8),
                Icon(Icons.place_outlined, size: 14, color: c.muted),
                const SizedBox(width: 2),
                Flexible(
                  child: Text(
                    post.place,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(color: c.muted, fontSize: 12.5),
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 10),
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: Text(post.title, style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600, height: 1.3)),
          ),
          const SizedBox(height: 6),
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: SelectableText(post.body, style: TextStyle(fontSize: 15, height: 1.5, color: c.text)),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Icon(Icons.verified_rounded, size: 16, color: c.afad),
              const SizedBox(width: 4),
              Flexible(
                child: Text(
                  post.orgName ?? '',
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                ),
              ),
              Text(
                ' · ${timeAgo(post.publishedAt ?? post.createdAt)}',
                style: TextStyle(color: c.muted, fontSize: 12.5),
              ),
            ],
          ),
          const SizedBox(height: 2),
          Row(
            children: [
              Icon(Icons.schedule_rounded, size: 14, color: c.muted),
              const SizedBox(width: 4),
              Expanded(
                child: Text(validLabel(post.validUntil), style: TextStyle(color: c.muted, fontSize: 12.5)),
              ),
              if (footer == null && onReport != null)
                TextButton.icon(
                  onPressed: onReport,
                  style: TextButton.styleFrom(foregroundColor: c.muted, visualDensity: VisualDensity.compact),
                  icon: const Icon(Icons.flag_outlined, size: 16),
                  label: const Text('Şikâyet et', style: TextStyle(fontSize: 12.5)),
                ),
            ],
          ),
          ?footer,
        ],
      ),
    );
  }
}

/// Searchable province list; returns '' for "all provinces" when [allowAll].
class ProvincePicker extends StatefulWidget {
  const ProvincePicker({super.key, this.allowAll = false});
  final bool allowAll;

  @override
  State<ProvincePicker> createState() => _ProvincePickerState();
}

class _ProvincePickerState extends State<ProvincePicker> {
  String _query = '';

  @override
  Widget build(BuildContext context) {
    final visible = provinces.where((p) => p.toLowerCase().contains(_query.toLowerCase())).toList();
    return SizedBox(
      height: MediaQuery.sizeOf(context).height * 0.75,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: TextField(
              autofocus: false,
              onChanged: (value) => setState(() => _query = value),
              decoration: InputDecoration(
                hintText: 'İl ara',
                prefixIcon: const Icon(Icons.search),
                isDense: true,
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
          Expanded(
            child: ListView(
              children: [
                if (widget.allowAll && _query.isEmpty)
                  ListTile(title: const Text('Tüm iller'), onTap: () => Navigator.pop(context, '')),
                for (final province in visible)
                  ListTile(title: Text(province), onTap: () => Navigator.pop(context, province)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
