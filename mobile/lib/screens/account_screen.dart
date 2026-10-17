import 'package:flutter/material.dart';

import '../api.dart';
import '../community.dart';
import '../theme.dart';
import 'admin_screen.dart';
import 'feed_screen.dart';

/// Organisations: log in or apply. Logged in: the account panel (write, my posts, administration).
class AccountScreen extends StatelessWidget {
  const AccountScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final session = CommunitySession.instance;
    return ListenableBuilder(
      listenable: session,
      builder: (context, _) => session.loggedIn ? _Panel(session: session) : const _LoginOrApply(),
    );
  }
}

class _LoginOrApply extends StatelessWidget {
  const _LoginOrApply();

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Kurum hesabı'),
          bottom: const TabBar(
            tabs: [
              Tab(text: 'Giriş'),
              Tab(text: 'Başvuru'),
            ],
          ),
        ),
        body: const TabBarView(children: [_LoginForm(), _ApplicationForm()]),
      ),
    );
  }
}

InputDecoration _decoration(String label, {String? hint, IconData? icon}) => InputDecoration(
  labelText: label,
  hintText: hint,
  prefixIcon: icon == null ? null : Icon(icon),
  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
);

void _snack(BuildContext context, String text) =>
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));

class _LoginForm extends StatefulWidget {
  const _LoginForm();

  @override
  State<_LoginForm> createState() => _LoginFormState();
}

class _LoginFormState extends State<_LoginForm> {
  final _email = TextEditingController();
  final _password = TextEditingController();
  bool _busy = false;
  bool _hidden = true;

  @override
  void dispose() {
    _email.dispose();
    _password.dispose();
    super.dispose();
  }

  Future<void> _login() async {
    setState(() => _busy = true);
    try {
      await CommunitySession.instance.login(_email.text, _password.text);
    } on ApiException catch (e) {
      if (mounted) _snack(context, e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 24, 20, 32),
      children: [
        Text(
          'Kurum hesapları duyuru paylaşabilir. Vatandaşların hesap açmasına gerek yoktur.',
          style: TextStyle(color: c.muted, height: 1.5),
        ),
        const SizedBox(height: 20),
        TextField(
          controller: _email,
          keyboardType: TextInputType.emailAddress,
          autofillHints: const [AutofillHints.email],
          decoration: _decoration('E-posta', icon: Icons.mail_outline),
        ),
        const SizedBox(height: 12),
        TextField(
          controller: _password,
          obscureText: _hidden,
          autofillHints: const [AutofillHints.password],
          onSubmitted: (_) => _login(),
          decoration: _decoration('Şifre', icon: Icons.lock_outline).copyWith(
            suffixIcon: IconButton(
              icon: Icon(_hidden ? Icons.visibility_outlined : Icons.visibility_off_outlined),
              onPressed: () => setState(() => _hidden = !_hidden),
            ),
          ),
        ),
        const SizedBox(height: 20),
        SizedBox(
          height: 50,
          child: FilledButton(
            onPressed: _busy ? null : _login,
            child: Text(_busy ? 'Giriş yapılıyor…' : 'Giriş yap', style: const TextStyle(fontSize: 16)),
          ),
        ),
      ],
    );
  }
}

class _ApplicationForm extends StatefulWidget {
  const _ApplicationForm();

  @override
  State<_ApplicationForm> createState() => _ApplicationFormState();
}

class _ApplicationFormState extends State<_ApplicationForm> {
  final _form = GlobalKey<FormState>();
  final _org = TextEditingController();
  final _contact = TextEditingController();
  final _email = TextEditingController();
  final _phone = TextEditingController();
  final _proof = TextEditingController();
  final _password = TextEditingController();
  String _type = 'belediye';
  bool _busy = false;

  @override
  void dispose() {
    for (final controller in [_org, _contact, _email, _phone, _proof, _password]) {
      controller.dispose();
    }
    super.dispose();
  }

  String? _required(String? value, {int min = 2}) =>
      (value ?? '').trim().length < min ? 'En az $min karakter olmalı' : null;

  Future<void> _submit() async {
    if (!_form.currentState!.validate()) return;
    setState(() => _busy = true);
    try {
      final message = await CommunitySession.instance.apply({
        'email': _email.text.trim(),
        'sifre': _password.text,
        'kurum_adi': _org.text.trim(),
        'kurum_turu': _type,
        'yetkili_adi': _contact.text.trim(),
        'telefon': _phone.text.trim(),
        'kanit': _proof.text.trim(),
      });
      if (!mounted) return;
      await showDialog<void>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Başvuru alındı'),
          content: Text(message),
          actions: [FilledButton(onPressed: () => Navigator.pop(context), child: const Text('Tamam'))],
        ),
      );
      if (mounted) Navigator.pop(context);
    } on ApiException catch (e) {
      if (mounted) _snack(context, e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Form(
      key: _form,
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 32),
        children: [
          Text(
            'Başvurunuzu yöneticiler inceler; kurumunuz doğrulandıktan sonra giriş yapabilirsiniz. Yeni kurumların '
            'duyuruları yayından önce kontrol edilir.',
            style: TextStyle(color: c.muted, height: 1.5),
          ),
          const SizedBox(height: 18),
          TextFormField(
            controller: _org,
            validator: _required,
            decoration: _decoration('Kurum adı', icon: Icons.apartment_outlined),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            initialValue: _type,
            decoration: _decoration('Kurum türü', icon: Icons.category_outlined),
            items: [for (final e in orgTypes.entries) DropdownMenuItem(value: e.key, child: Text(e.value))],
            onChanged: (value) => setState(() => _type = value ?? _type),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _contact,
            validator: _required,
            decoration: _decoration('Yetkili adı soyadı', icon: Icons.person_outline),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _email,
            keyboardType: TextInputType.emailAddress,
            validator: (v) =>
                RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch((v ?? '').trim()) ? null : 'Geçerli bir e-posta girin',
            decoration: _decoration('Kurumsal e-posta', icon: Icons.mail_outline),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _phone,
            keyboardType: TextInputType.phone,
            decoration: _decoration('Telefon', icon: Icons.phone_outlined),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _proof,
            minLines: 2,
            maxLines: 4,
            validator: (v) => _required(v, min: 5),
            decoration: _decoration(
              'Kurumla bağınızı gösteren bilgi',
              hint: 'Resmî web sayfası, görev yeri, kurum e-postası, telefonla doğrulanabilecek birim…',
              icon: Icons.fact_check_outlined,
            ),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _password,
            obscureText: true,
            validator: (v) => (v ?? '').length < 8 ? 'Şifre en az 8 karakter olmalı' : null,
            decoration: _decoration('Şifre', icon: Icons.lock_outline),
          ),
          const SizedBox(height: 20),
          SizedBox(
            height: 50,
            child: FilledButton(
              onPressed: _busy ? null : _submit,
              child: Text(_busy ? 'Gönderiliyor…' : 'Başvuruyu gönder', style: const TextStyle(fontSize: 16)),
            ),
          ),
        ],
      ),
    );
  }
}

class _Panel extends StatelessWidget {
  const _Panel({required this.session});
  final CommunitySession session;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final account = session.account!;
    void go(Widget screen) => Navigator.push(context, MaterialPageRoute(builder: (_) => screen));
    return Scaffold(
      appBar: AppBar(title: Text(account.isAdmin ? 'Yönetici hesabı' : 'Kurum hesabı')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
        children: [
          ListTile(
            contentPadding: EdgeInsets.zero,
            leading: CircleAvatar(
              backgroundColor: c.surfaceHigh,
              child: Icon(account.isAdmin ? Icons.admin_panel_settings_outlined : Icons.apartment_outlined),
            ),
            title: Row(
              children: [
                Flexible(
                  child: Text(account.orgName, style: const TextStyle(fontWeight: FontWeight.w600)),
                ),
                const SizedBox(width: 6),
                Icon(Icons.verified_rounded, size: 18, color: c.afad),
              ],
            ),
            subtitle: Text(
              '${account.contactName} · ${account.email}\n'
              '${account.isAdmin
                  ? 'Yönetici'
                  : account.trusted
                  ? 'Güvenilir kurum: duyurular doğrudan yayınlanır'
                  : 'Duyurular yönetici onayından sonra yayınlanır'}',
              style: TextStyle(color: c.muted, height: 1.4),
            ),
            isThreeLine: true,
          ),
          const SizedBox(height: 12),
          _action(context, Icons.edit_outlined, 'Duyuru yaz', () => go(const ComposePostScreen())),
          _action(context, Icons.list_alt_outlined, 'Duyurularım', () => go(const MyPostsScreen())),
          if (account.isAdmin)
            _action(context, Icons.admin_panel_settings_outlined, 'Yönetim', () => go(const AdminScreen())),
          const SizedBox(height: 20),
          OutlinedButton.icon(
            onPressed: () async {
              await session.logout();
              if (context.mounted) Navigator.pop(context);
            },
            icon: const Icon(Icons.logout),
            label: const Text('Çıkış yap'),
          ),
        ],
      ),
    );
  }

  Widget _action(BuildContext context, IconData icon, String label, VoidCallback onTap) {
    final c = context.colors;
    return Card(
      color: c.surface,
      elevation: 0,
      margin: const EdgeInsets.only(bottom: 10),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: BorderSide(color: c.border),
      ),
      child: ListTile(
        leading: Icon(icon, color: c.accent),
        title: Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }
}

class ComposePostScreen extends StatefulWidget {
  const ComposePostScreen({super.key});

  @override
  State<ComposePostScreen> createState() => _ComposePostScreenState();
}

class _ComposePostScreenState extends State<ComposePostScreen> {
  static const _validity = [(6, '6 saat'), (24, '1 gün'), (72, '3 gün'), (168, '1 hafta'), (720, '30 gün')];

  final _form = GlobalKey<FormState>();
  final _title = TextEditingController();
  final _body = TextEditingController();
  final _district = TextEditingController();
  String _kind = 'duyuru';
  String? _province;
  int _hours = 72;
  bool _busy = false;

  @override
  void dispose() {
    _title.dispose();
    _body.dispose();
    _district.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_form.currentState!.validate()) return;
    setState(() => _busy = true);
    try {
      final message = await CommunitySession.instance.createPost({
        'baslik': _title.text.trim(),
        'metin': _body.text.trim(),
        'tur': _kind,
        'il': _province,
        'ilce': _district.text.trim().isEmpty ? null : _district.text.trim(),
        'gecerlilik_saat': _hours,
      });
      if (!mounted) return;
      _snack(context, message);
      Navigator.pop(context);
    } on ApiException catch (e) {
      if (mounted) _snack(context, e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final account = CommunitySession.instance.account;
    return Scaffold(
      appBar: AppBar(title: const Text('Duyuru yaz')),
      body: Form(
        key: _form,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
          children: [
            Text('Tür', style: TextStyle(color: c.muted, fontSize: 13)),
            const SizedBox(height: 6),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final e in postKinds.entries)
                  ChoiceChip(
                    avatar: Icon(kindIcon(e.key), size: 16),
                    label: Text(e.value),
                    selected: _kind == e.key,
                    onSelected: (_) => setState(() => _kind = e.key),
                  ),
              ],
            ),
            const SizedBox(height: 16),
            TextFormField(
              controller: _title,
              maxLength: 120,
              validator: (v) => (v ?? '').trim().length < 3 ? 'Başlık en az 3 karakter olmalı' : null,
              decoration: _decoration('Başlık', hint: 'Ör. Defne parkında su dağıtımı'),
            ),
            const SizedBox(height: 4),
            TextFormField(
              controller: _body,
              minLines: 4,
              maxLines: 10,
              maxLength: 2000,
              validator: (v) => (v ?? '').trim().length < 5 ? 'Metin en az 5 karakter olmalı' : null,
              decoration: _decoration('Metin', hint: 'Ne, nerede, ne zaman? Açık adres ve saat yazın.'),
            ),
            const SizedBox(height: 4),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () async {
                      final picked = await showModalBottomSheet<String>(
                        context: context,
                        isScrollControlled: true,
                        showDragHandle: true,
                        builder: (_) => const ProvincePicker(allowAll: true),
                      );
                      if (picked != null) setState(() => _province = picked.isEmpty ? null : picked);
                    },
                    icon: const Icon(Icons.place_outlined),
                    label: Text(_province ?? 'Tüm Türkiye'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: TextFormField(
                    controller: _district,
                    enabled: _province != null,
                    decoration: InputDecoration(
                      labelText: 'İlçe (isteğe bağlı)',
                      isDense: true,
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text('Ne kadar süre geçerli?', style: TextStyle(color: c.muted, fontSize: 13)),
            const SizedBox(height: 6),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final (hours, label) in _validity)
                  ChoiceChip(
                    label: Text(label),
                    selected: _hours == hours,
                    onSelected: (_) => setState(() => _hours = hours),
                  ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              'Süresi dolan duyuru akıştan kalkar; afette eski bilgi de yanlış bilgi kadar tehlikelidir.',
              style: TextStyle(color: c.muted, fontSize: 12.5, height: 1.4),
            ),
            const SizedBox(height: 20),
            SizedBox(
              height: 50,
              child: FilledButton.icon(
                onPressed: _busy ? null : _submit,
                icon: const Icon(Icons.send_rounded),
                label: Text(
                  _busy
                      ? 'Gönderiliyor…'
                      : (account?.trusted ?? false)
                      ? 'Yayınla'
                      : 'Onaya gönder',
                  style: const TextStyle(fontSize: 16),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

const statusNames = {
  'beklemede': 'Onay bekliyor',
  'yayinda': 'Yayında',
  'reddedildi': 'Reddedildi',
  'kaldirildi': 'Kaldırıldı',
  'gizlendi': 'Şikâyet nedeniyle gizlendi',
};

Color statusColor(DargPalette c, String status) => switch (status) {
  'yayinda' => c.afad,
  'beklemede' => c.accent,
  _ => c.emergency,
};

class MyPostsScreen extends StatefulWidget {
  const MyPostsScreen({super.key});

  @override
  State<MyPostsScreen> createState() => _MyPostsScreenState();
}

class _MyPostsScreenState extends State<MyPostsScreen> {
  List<Post>? _posts;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final posts = await CommunitySession.instance.myPosts();
      if (mounted) setState(() => _posts = posts);
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  Future<void> _delete(Post post) async {
    try {
      await CommunitySession.instance.deletePost(post.id);
      _load();
    } on ApiException catch (e) {
      if (mounted) _snack(context, e.message);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Scaffold(
      appBar: AppBar(title: const Text('Duyurularım')),
      body: _error != null
          ? Center(child: Text(_error!))
          : _posts == null
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_posts!.isEmpty)
                    Padding(
                      padding: const EdgeInsets.all(32),
                      child: Text(
                        'Henüz duyuru yazmadınız.',
                        textAlign: TextAlign.center,
                        style: TextStyle(color: c.muted),
                      ),
                    ),
                  for (final post in _posts!)
                    PostCard(
                      post: post,
                      footer: Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                            decoration: BoxDecoration(
                              color: statusColor(c, post.status).withValues(alpha: 0.14),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              statusNames[post.status] ?? post.status,
                              style: TextStyle(
                                color: statusColor(c, post.status),
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                          if (post.reviewNote != null && post.reviewNote!.isNotEmpty) ...[
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(post.reviewNote!, style: TextStyle(color: c.muted, fontSize: 12.5)),
                            ),
                          ] else
                            const Spacer(),
                          if (post.status == 'yayinda' || post.status == 'beklemede')
                            TextButton(onPressed: () => _delete(post), child: const Text('Kaldır')),
                        ],
                      ),
                    ),
                ],
              ),
            ),
    );
  }
}
