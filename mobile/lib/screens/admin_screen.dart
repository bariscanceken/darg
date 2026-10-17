import 'package:flutter/material.dart';

import '../api.dart';
import '../community.dart';
import '../theme.dart';
import 'feed_screen.dart';

/// Moderation for administrators: pending posts, reported posts, organisation applications, organisations.
class AdminScreen extends StatefulWidget {
  const AdminScreen({super.key});

  @override
  State<AdminScreen> createState() => _AdminScreenState();
}

class _AdminScreenState extends State<AdminScreen> {
  final _session = CommunitySession.instance;
  Map<String, dynamic>? _summary;
  int _version = 0; // bumped after every decision so the tabs reload

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    try {
      final summary = await _session.adminSummary();
      if (mounted) {
        setState(() {
          _summary = summary;
          _version++;
        });
      }
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  Future<void> run(Future<void> Function() action, String done) async {
    try {
      await action();
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(done)));
      await _refresh();
    } on ApiException catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
    }
  }

  Future<String?> askNote(String title) {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: TextField(
          controller: controller,
          maxLines: 3,
          maxLength: 500,
          decoration: const InputDecoration(hintText: 'Gerekçe (kuruma gösterilir)', border: OutlineInputBorder()),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Vazgeç')),
          FilledButton(onPressed: () => Navigator.pop(context, controller.text), child: const Text('Tamam')),
        ],
      ),
    );
  }

  String _tab(String label, String key) {
    final n = _summary?[key] as int? ?? 0;
    return n > 0 ? '$label ($n)' : label;
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 4,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Yönetim'),
          bottom: TabBar(
            isScrollable: true,
            tabAlignment: TabAlignment.start,
            tabs: [
              Tab(text: _tab('Onay kuyruğu', 'bekleyen_gonderi')),
              Tab(text: _tab('Şikâyetler', 'sikayetli_gonderi')),
              Tab(text: _tab('Başvurular', 'bekleyen_basvuru')),
              Tab(text: _tab('Kurumlar', 'aktif_kurum')),
            ],
          ),
        ),
        body: TabBarView(
          children: [
            _PostQueue(key: ValueKey('q$_version'), status: 'beklemede', admin: this),
            _PostQueue(key: ValueKey('r$_version'), status: 'sikayet', admin: this),
            _Organisations(key: ValueKey('a$_version'), status: 'beklemede', admin: this),
            _Organisations(key: ValueKey('o$_version'), status: 'aktif', admin: this),
          ],
        ),
      ),
    );
  }
}

class _PostQueue extends StatelessWidget {
  const _PostQueue({super.key, required this.status, required this.admin});
  final String status;
  final _AdminScreenState admin;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final session = CommunitySession.instance;
    final reports = status == 'sikayet'; // every post with open reports, published or hidden
    return FutureBuilder<List<Post>>(
      future: reports ? session.adminReported() : session.adminPosts(status),
      builder: (context, snapshot) {
        if (snapshot.hasError) return Center(child: Text('${snapshot.error}'));
        if (!snapshot.hasData) return const Center(child: CircularProgressIndicator());
        final posts = snapshot.data!;
        if (posts.isEmpty) {
          return Center(
            child: Text(
              reports ? 'Şikâyet edilen duyuru yok.' : 'Onay bekleyen duyuru yok.',
              style: TextStyle(color: c.muted),
            ),
          );
        }
        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            for (final post in posts)
              PostCard(
                post: post,
                footer: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (reports)
                      Padding(
                        padding: const EdgeInsets.only(top: 6, right: 8),
                        child: Text(
                          '${post.reports} şikâyet · '
                          '${post.status == 'gizlendi' ? 'gizli' : 'hâlâ yayında, kararınızı bekliyor'}'
                          '${post.reportReasons.isEmpty ? '' : '\n${post.reportReasons.map((r) => '“$r”').join(' · ')}'}',
                          style: TextStyle(color: c.emergency, fontSize: 13, height: 1.4),
                        ),
                      ),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        TextButton(
                          onPressed: () async {
                            final note = await admin.askNote(reports ? 'Kaldır' : 'Reddet');
                            if (note == null) return;
                            admin.run(
                              () => session.adminDecidePost(post.id, reports ? 'kaldir' : 'reddet', note: note),
                              reports ? 'Duyuru kaldırıldı' : 'Duyuru reddedildi',
                            );
                          },
                          style: TextButton.styleFrom(foregroundColor: c.emergency),
                          child: Text(reports ? 'Kaldır' : 'Reddet'),
                        ),
                        const SizedBox(width: 4),
                        FilledButton(
                          onPressed: () => admin.run(
                            () => session.adminDecidePost(post.id, 'yayinla'),
                            !reports
                                ? 'Duyuru yayınlandı'
                                : post.status == 'gizlendi'
                                ? 'Duyuru yeniden yayında'
                                : 'Şikâyetler kapatıldı',
                          ),
                          child: Text(
                            !reports
                                ? 'Yayınla'
                                : post.status == 'gizlendi'
                                ? 'Yeniden yayınla'
                                : 'Şikâyeti kapat',
                          ),
                        ),
                        const SizedBox(width: 8),
                      ],
                    ),
                  ],
                ),
              ),
          ],
        );
      },
    );
  }
}

class _Organisations extends StatelessWidget {
  const _Organisations({super.key, required this.status, required this.admin});
  final String status;
  final _AdminScreenState admin;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final session = CommunitySession.instance;
    return FutureBuilder<List<Account>>(
      future: session.adminOrganisations(status: status),
      builder: (context, snapshot) {
        if (snapshot.hasError) return Center(child: Text('${snapshot.error}'));
        if (!snapshot.hasData) return const Center(child: CircularProgressIndicator());
        final organisations = snapshot.data!;
        if (organisations.isEmpty) {
          return Center(
            child: Text(
              status == 'beklemede' ? 'Bekleyen başvuru yok.' : 'Henüz onaylı kurum yok.',
              style: TextStyle(color: c.muted),
            ),
          );
        }
        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            for (final org in organisations)
              Container(
                margin: const EdgeInsets.only(bottom: 12),
                padding: const EdgeInsets.fromLTRB(14, 12, 8, 8),
                decoration: BoxDecoration(
                  color: c.surface,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: c.border),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(org.orgName, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
                    const SizedBox(height: 4),
                    Text(
                      '${orgTypes[org.orgType] ?? org.orgType} · ${org.contactName}\n${org.email}'
                      '${org.phone.isEmpty ? '' : ' · ${org.phone}'}',
                      style: TextStyle(color: c.muted, height: 1.4),
                    ),
                    if (org.proof.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Text('Kanıt', style: TextStyle(color: c.muted, fontSize: 12)),
                      SelectableText(org.proof, style: const TextStyle(height: 1.4)),
                    ],
                    const SizedBox(height: 6),
                    if (status == 'beklemede')
                      Row(
                        mainAxisAlignment: MainAxisAlignment.end,
                        children: [
                          TextButton(
                            onPressed: () async {
                              final note = await admin.askNote('Başvuruyu reddet');
                              if (note == null) return;
                              admin.run(
                                () => session.adminDecideOrganisation(org.id, 'reddet', note: note),
                                'Başvuru reddedildi',
                              );
                            },
                            style: TextButton.styleFrom(foregroundColor: c.emergency),
                            child: const Text('Reddet'),
                          ),
                          OutlinedButton(
                            onPressed: () => admin.run(
                              () => session.adminDecideOrganisation(org.id, 'onayla', trusted: true),
                              'Kurum güvenilir olarak onaylandı',
                            ),
                            child: const Text('Güvenilir onayla'),
                          ),
                          const SizedBox(width: 6),
                          FilledButton(
                            onPressed: () =>
                                admin.run(() => session.adminDecideOrganisation(org.id, 'onayla'), 'Kurum onaylandı'),
                            child: const Text('Onayla'),
                          ),
                        ],
                      )
                    else
                      Row(
                        children: [
                          Expanded(
                            child: SwitchListTile(
                              contentPadding: EdgeInsets.zero,
                              dense: true,
                              title: const Text('Güvenilir'),
                              subtitle: const Text('Duyuruları onaysız yayınlanır'),
                              value: org.trusted,
                              onChanged: (value) => admin.run(
                                () => session.adminTrust(org.id, value),
                                value ? 'Kurum güvenilir yapıldı' : 'Kurumun duyuruları artık onaya düşecek',
                              ),
                            ),
                          ),
                          TextButton(
                            onPressed: () async {
                              final note = await admin.askNote('Hesabı askıya al');
                              if (note == null) return;
                              admin.run(
                                () => session.adminDecideOrganisation(org.id, 'askiya_al', note: note),
                                'Hesap askıya alındı',
                              );
                            },
                            style: TextButton.styleFrom(foregroundColor: c.emergency),
                            child: const Text('Askıya al'),
                          ),
                        ],
                      ),
                  ],
                ),
              ),
          ],
        );
      },
    );
  }
}
