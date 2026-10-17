import 'package:flutter/material.dart';

import '../api.dart';
import '../storage.dart';
import '../theme.dart';

class AboutScreen extends StatefulWidget {
  const AboutScreen({super.key, required this.storage});
  final Storage storage;

  @override
  State<AboutScreen> createState() => _AboutScreenState();
}

class _AboutScreenState extends State<AboutScreen> {
  late final _server = TextEditingController(text: widget.storage.serverUrl);
  String? _status;

  @override
  void dispose() {
    _server.dispose();
    super.dispose();
  }

  Future<void> _saveAndTest() async {
    await widget.storage.setServerUrl(_server.text);
    setState(() => _status = 'Bağlantı deneniyor…');
    final ok = await DargApi(widget.storage.serverUrl).health();
    if (mounted) setState(() => _status = ok ? 'Sunucuya bağlanıldı ✓' : 'Sunucuya ulaşılamadı');
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final body = TextStyle(fontSize: 15, height: 1.55);
    final heading = TextStyle(fontSize: 17, fontWeight: FontWeight.w600);
    return Scaffold(
      appBar: AppBar(title: Text('Hakkında')),
      body: ListView(
        padding: EdgeInsets.fromLTRB(20, 8, 20, 32),
        children: [
          Text('DARG', style: TextStyle(fontSize: 28, fontWeight: FontWeight.w600, letterSpacing: 2)),
          SizedBox(height: 4),
          Text('Deprem bilgi asistanı', style: TextStyle(color: c.muted)),
          SizedBox(height: 18),
          Text(
            'DARG, depremle ilgili sorularınızı AFAD belgelerine dayanarak cevaplar. AFAD belgelerinde bulunmayan '
            'konularda genel bilgi ve resmî web siteleri (gov.tr, Kızılay) kullanılır. Her cümlenin neye dayandığı '
            'cevapta gösterilir.',
            style: body,
          ),
          SizedBox(height: 22),
          Text('Kaynak işaretleri', style: heading),
          SizedBox(height: 10),
          _legendRow(c.afad, '1', 'AFAD belgesinden alıntı (sayfa numarasıyla)'),
          _legendRow(c.web, '2', 'Resmî web sitesi (yayın tarihi farklı olabilir)'),
          _legendRow(c.general, 'genel bilgi', 'AFAD belgelerinde doğrulanmamış cümle'),
          SizedBox(height: 22),
          ValueListenableBuilder<ThemeMode>(
            valueListenable: widget.storage.themeMode,
            builder: (context, mode, _) => SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: Text('Koyu tema', style: heading),
              value: mode == ThemeMode.dark,
              onChanged: (_) => widget.storage.toggleTheme(),
            ),
          ),
          SizedBox(height: 14),
          Text('Önemli', style: heading),
          SizedBox(height: 8),
          Text(
            'DARG bir bilgi asistanıdır; 112, AFAD ya da sağlık profesyonellerinin yerini tutmaz. Acil bir '
            'durumdaysanız önce 112\'yi arayın.\n\n'
            'Sorularınız ve geri bildirimleriniz, sistemi iyileştirmek amacıyla sunucuda kaydedilir. Sorularınıza '
            'ad, telefon ya da adres gibi kişisel bilgiler yazmayın. Sohbet geçmişi yalnızca bu cihazda tutulur.',
            style: body,
          ),
          SizedBox(height: 22),
          Text('Sunucu adresi', style: heading),
          SizedBox(height: 8),
          TextField(
            controller: _server,
            keyboardType: TextInputType.url,
            decoration: InputDecoration(border: OutlineInputBorder(), hintText: 'http://192.168.1.20:8000'),
          ),
          SizedBox(height: 10),
          Row(
            children: [
              FilledButton(onPressed: _saveAndTest, child: Text('Kaydet ve dene')),
              SizedBox(width: 12),
              if (_status != null)
                Expanded(
                  child: Text(_status!, style: TextStyle(color: c.muted)),
                ),
            ],
          ),
          SizedBox(height: 28),
          Text(
            'TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projesi · Karadeniz Teknik Üniversitesi',
            style: TextStyle(color: c.muted, fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _legendRow(Color color, String label, String text) => Padding(
    padding: EdgeInsets.only(bottom: 8),
    child: Row(
      children: [
        Container(
          width: 84,
          alignment: Alignment.centerLeft,
          child: Container(
            padding: EdgeInsets.symmetric(horizontal: 6, vertical: 1),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.14),
              borderRadius: BorderRadius.circular(6),
              border: Border.all(color: color.withValues(alpha: 0.35)),
            ),
            child: Text(
              label,
              style: TextStyle(fontSize: 12, color: color, fontWeight: FontWeight.w600),
            ),
          ),
        ),
        Expanded(child: Text(text, style: TextStyle(fontSize: 14))),
      ],
    ),
  );
}
