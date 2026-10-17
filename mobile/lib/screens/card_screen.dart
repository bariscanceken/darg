import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../emergency_card.dart';
import '../storage.dart';
import '../theme.dart';
import 'chat_screen.dart';

/// Fill in or view the emergency card. On first launch it is the welcome screen (and can be skipped).
class CardScreen extends StatefulWidget {
  const CardScreen({super.key, required this.storage, required this.cards, this.firstLaunch = false});

  final Storage storage;
  final EmergencyCardStore cards;
  final bool firstLaunch;

  @override
  State<CardScreen> createState() => _CardScreenState();
}

class _CardScreenState extends State<CardScreen> {
  static const _maxContacts = 3;

  final _name = TextEditingController();
  final _health = TextEditingController();
  final _address = TextEditingController();
  final _contacts = <(TextEditingController, TextEditingController, TextEditingController)>[];
  String _bloodType = '';
  bool _lockScreen = false;
  bool _loading = true;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    widget.cards.load().then((card) {
      if (!mounted) return;
      setState(() {
        _name.text = card.fullName;
        _health.text = card.health;
        _address.text = card.address;
        _bloodType = card.bloodType;
        _lockScreen = card.showOnLockScreen;
        for (final contact in card.contacts) {
          _addContact(contact);
        }
        while (_contacts.length < 2) {
          _addContact();
        }
        _loading = false;
      });
    });
  }

  void _addContact([EmergencyContact? contact]) {
    _contacts.add((
      TextEditingController(text: contact?.name),
      TextEditingController(text: contact?.relation),
      TextEditingController(text: contact?.phone),
    ));
  }

  @override
  void dispose() {
    for (final controller in [
      _name,
      _health,
      _address,
      for (final c in _contacts) ...[c.$1, c.$2, c.$3],
    ]) {
      controller.dispose();
    }
    super.dispose();
  }

  EmergencyCard get _card => EmergencyCard(
    fullName: _name.text,
    bloodType: _bloodType,
    health: _health.text,
    address: _address.text,
    contacts: [
      for (final c in _contacts) EmergencyContact(name: c.$1.text, relation: c.$2.text, phone: c.$3.text),
    ].where((c) => !c.isEmpty).toList(),
    showOnLockScreen: _lockScreen,
  );

  Future<void> _save() async {
    setState(() => _saving = true);
    final shown = await widget.cards.save(_card);
    if (!mounted) return;
    setState(() => _saving = false);
    if (!shown) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Kart kaydedildi, ama bildirim izni verilmediği için kilit ekranında görünmeyecek.'),
        ),
      );
    } else if (!widget.firstLaunch) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Acil durum kartı kaydedildi')));
    }
    _finish();
  }

  Future<void> _finish() async {
    if (!widget.firstLaunch) {
      Navigator.pop(context);
      return;
    }
    await widget.storage.setWelcomed();
    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => ChatScreen(storage: widget.storage, cards: widget.cards),
      ),
    );
  }

  Future<void> _delete() async {
    final sure = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Kart silinsin mi?'),
        content: const Text('Acil durum kartındaki tüm bilgiler bu telefondan silinir.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Vazgeç')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Sil')),
        ],
      ),
    );
    if (sure != true) return;
    await widget.cards.delete();
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Scaffold(
      appBar: AppBar(
        automaticallyImplyLeading: !widget.firstLaunch,
        title: Text(widget.firstLaunch ? 'Hoş geldiniz' : 'Acil durum kartım'),
        actions: [
          if (!widget.firstLaunch && !_loading) ...[
            IconButton(
              tooltip: 'Kurtarıcıya göster',
              icon: const Icon(Icons.badge_outlined),
              onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => CardViewScreen(card: _card))),
            ),
            IconButton(tooltip: 'Kartı sil', icon: const Icon(Icons.delete_outline), onPressed: _delete),
          ],
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.fromLTRB(18, 8, 18, 32),
              children: [
                if (widget.firstLaunch) ...[
                  Icon(Icons.badge_outlined, size: 40, color: c.accent),
                  const SizedBox(height: 12),
                  const Text(
                    'Acil durum kartınızı oluşturun',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 8),
                ],
                Text(
                  'Bir afette telefonunuz bulunursa ya da kendinizi ifade edemezseniz, kurtarma ekipleri bu bilgilere '
                  'ulaşabilir. Bilgiler yalnızca bu telefonda şifreli saklanır, sunucuya gönderilmez. Tüm alanlar '
                  'isteğe bağlıdır.',
                  style: TextStyle(color: c.muted, height: 1.5),
                ),
                const SizedBox(height: 20),
                _field(_name, 'Ad soyad', icon: Icons.person_outline),
                const SizedBox(height: 12),
                DropdownButtonFormField<String>(
                  initialValue: _bloodType.isEmpty ? null : _bloodType,
                  decoration: _decoration('Kan grubu', Icons.bloodtype_outlined),
                  items: [
                    for (final type in EmergencyCard.bloodTypes) DropdownMenuItem(value: type, child: Text(type)),
                    const DropdownMenuItem(value: '', child: Text('Bilmiyorum')),
                  ],
                  onChanged: (value) => setState(() => _bloodType = value ?? ''),
                ),
                const SizedBox(height: 12),
                _field(
                  _health,
                  'Sağlık bilgisi',
                  icon: Icons.medical_information_outlined,
                  hint: 'Kronik hastalık, alerji, kullandığınız ilaçlar',
                  lines: 2,
                ),
                const SizedBox(height: 12),
                _field(_address, 'Adres', icon: Icons.home_outlined, lines: 2),
                const SizedBox(height: 24),
                const Text('Yakınlarınız', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text('Size ulaşılamazsa aranacak kişiler', style: TextStyle(color: c.muted, fontSize: 13)),
                const SizedBox(height: 12),
                for (final (index, contact) in _contacts.indexed) ...[
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: c.surface,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: c.border),
                    ),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            Expanded(flex: 3, child: _field(contact.$1, '${index + 1}. yakın adı', dense: true)),
                            const SizedBox(width: 8),
                            Expanded(flex: 2, child: _field(contact.$2, 'Yakınlık', hint: 'Anne, eş…', dense: true)),
                          ],
                        ),
                        const SizedBox(height: 8),
                        _field(contact.$3, 'Telefon', icon: Icons.phone_outlined, phone: true, dense: true),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                ],
                if (_contacts.length < _maxContacts)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: TextButton.icon(
                      onPressed: () => setState(_addContact),
                      icon: const Icon(Icons.add),
                      label: const Text('Yakın ekle'),
                    ),
                  ),
                if (EmergencyCardStore.lockScreenSupported) ...[
                  const SizedBox(height: 12),
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    value: _lockScreen,
                    onChanged: (value) => setState(() => _lockScreen = value),
                    title: const Text('Kilit ekranında göster'),
                    subtitle: Text(
                      'Telefonu bulan kişi kartı kilidi açmadan görebilir. Kilit ekranını gören herkes bu bilgileri '
                      'okuyabilir.',
                      style: TextStyle(color: c.muted, fontSize: 13),
                    ),
                  ),
                ],
                const SizedBox(height: 8),
                Text(
                  'Ayrıca telefonunuzun kendi "Acil durum bilgileri" (iPhone\'da "Tıbbi Kimlik") bölümünü de '
                  'doldurmanızı öneririz; kurtarma ekipleri önce oraya bakar.',
                  style: TextStyle(color: c.muted, fontSize: 13, height: 1.4),
                ),
                const SizedBox(height: 24),
                SizedBox(
                  height: 50,
                  child: FilledButton(
                    onPressed: _saving ? null : _save,
                    child: Text(_saving ? 'Kaydediliyor…' : 'Kaydet', style: const TextStyle(fontSize: 16)),
                  ),
                ),
                if (widget.firstLaunch) ...[
                  const SizedBox(height: 8),
                  TextButton(onPressed: _finish, child: const Text('Şimdilik geç')),
                ],
              ],
            ),
    );
  }

  InputDecoration _decoration(String label, IconData? icon, {String? hint, bool dense = false}) => InputDecoration(
    labelText: label,
    hintText: hint,
    prefixIcon: icon == null ? null : Icon(icon),
    isDense: dense,
    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
  );

  Widget _field(
    TextEditingController controller,
    String label, {
    IconData? icon,
    String? hint,
    int lines = 1,
    bool phone = false,
    bool dense = false,
  }) => TextField(
    controller: controller,
    minLines: lines,
    maxLines: lines,
    keyboardType: phone ? TextInputType.phone : (lines > 1 ? TextInputType.multiline : TextInputType.text),
    textCapitalization: phone ? TextCapitalization.none : TextCapitalization.words,
    decoration: _decoration(label, icon, hint: hint, dense: dense),
  );
}

/// Large, high-contrast view for a rescuer, with call buttons for the contacts.
class CardViewScreen extends StatelessWidget {
  const CardViewScreen({super.key, required this.card});
  final EmergencyCard card;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    Widget row(String label, String value) => Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: TextStyle(color: c.muted, fontSize: 14)),
          const SizedBox(height: 2),
          Text(value, style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w600, height: 1.3)),
        ],
      ),
    );
    return Scaffold(
      appBar: AppBar(title: const Text('ACİL DURUM KARTI')),
      body: card.isEmpty
          ? const Center(child: Text('Kart boş'))
          : ListView(
              padding: const EdgeInsets.fromLTRB(22, 8, 22, 32),
              children: [
                if (card.fullName.trim().isNotEmpty) row('Ad soyad', card.fullName.trim()),
                if (card.bloodType.isNotEmpty) row('Kan grubu', card.bloodType),
                if (card.health.trim().isNotEmpty) row('Sağlık bilgisi', card.health.trim()),
                if (card.address.trim().isNotEmpty) row('Adres', card.address.trim()),
                for (final contact in card.contacts.where((x) => !x.isEmpty))
                  Card(
                    color: c.surface,
                    margin: const EdgeInsets.only(bottom: 10),
                    child: ListTile(
                      title: Text(
                        contact.name.trim() + (contact.relation.trim().isEmpty ? '' : ' (${contact.relation.trim()})'),
                        style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                      ),
                      subtitle: Text(contact.phone.trim(), style: const TextStyle(fontSize: 16)),
                      trailing: contact.phone.trim().isEmpty
                          ? null
                          : IconButton.filled(
                              onPressed: () => launchUrl(Uri.parse('tel:${contact.phone.replaceAll(' ', '')}')),
                              icon: const Icon(Icons.phone),
                              tooltip: 'Ara',
                            ),
                    ),
                  ),
              ],
            ),
    );
  }
}
