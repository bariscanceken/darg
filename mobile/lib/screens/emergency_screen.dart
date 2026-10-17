import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../theme.dart';
import 'signal_screen.dart';

/// Built into the app (no server, no internet): networks often fail right after an earthquake.
/// Content: the AFAD citizen guides' core advice ("Deprem Anında", "İlk 72 Saat").
class EmergencyScreen extends StatelessWidget {
  const EmergencyScreen({super.key});

  static const _sections = [
    (
      Icons.vibration,
      'Deprem sırasında',
      [
        'Çök, Kapan, Tutun: sağlam bir masanın yanına çökün, başınızı ve boynunuzu koruyun, tutunun.',
        'Pencerelerden, camlardan ve devrilebilecek eşyalardan uzak durun.',
        'Asansör ve merdiven kullanmayın; dışarı çıkmaya çalışmayın.',
        'Dışarıdaysanız binalardan, direklerden ve elektrik tellerinden uzak, açık bir alana geçin.',
        'Araçtaysanız güvenli bir yerde durun, köprü ve tünel üzerinde durmayın, araçta kalın.',
      ],
    ),
    (
      Icons.warning_amber_rounded,
      'Enkaz altındaysanız',
      [
        'Sakin olun, gereksiz hareketle toz kaldırmayın; ağzınızı ve burnunuzu bir bezle kapatın.',
        'Boru ya da duvara düzenli aralıklarla vurarak ses çıkarın; varsa düdük kullanın.',
        'Bağırmayı son çare olarak kullanın; enerjinizi koruyun.',
        'Telefonunuz çalışıyorsa 112\'yi arayın ya da mesaj atın.',
      ],
    ),
    (
      Icons.home_work_outlined,
      'Deprem sonrası',
      [
        'Önce kendinizi, sonra çevrenizdekileri kontrol edin; yaralı varsa 112\'yi arayın.',
        'Gaz kokusu varsa kibrit, çakmak ve elektrik düğmesi kullanmayın; vanayı kapatın.',
        'Hasarlı binalara girmeyin; artçı sarsıntılara hazırlıklı olun.',
        'En yakın toplanma alanına gidin. Toplanma alanınızı e-Devlet\'te "AFAD Acil Toplanma Alanı '
            'Sorgulama" hizmetinden öğrenebilirsiniz.',
        'Telefon hatlarını acil durumlar için meşgul etmeyin; mümkünse mesaj kullanın.',
      ],
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Scaffold(
      appBar: AppBar(title: Text('Acil durum bilgileri')),
      body: ListView(
        padding: EdgeInsets.fromLTRB(18, 8, 18, 32),
        children: [
          SizedBox(
            height: 64,
            child: FilledButton.icon(
              onPressed: () => launchUrl(Uri.parse('tel:112')),
              style: FilledButton.styleFrom(
                backgroundColor: c.emergency,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              ),
              icon: Icon(Icons.phone_in_talk, size: 26),
              label: Text('112 Acil Çağrı', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700)),
            ),
          ),
          SizedBox(height: 8),
          Text(
            'Ambulans, itfaiye, polis, jandarma ve AFAD için tek numara.',
            textAlign: TextAlign.center,
            style: TextStyle(color: c.muted, fontSize: 13),
          ),
          const SizedBox(height: 14),
          SizedBox(
            height: 52,
            child: OutlinedButton.icon(
              onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const SignalScreen())),
              style: OutlinedButton.styleFrom(
                foregroundColor: c.emergency,
                side: BorderSide(color: c.emergency, width: 1.5),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              ),
              icon: const Icon(Icons.campaign_outlined),
              label: const Text('Yardım sinyali: düdük, SOS feneri', style: TextStyle(fontWeight: FontWeight.w600)),
            ),
          ),
          for (final (icon, title, items) in _sections) ...[
            SizedBox(height: 24),
            Row(
              children: [
                Icon(icon, color: c.accent),
                SizedBox(width: 10),
                Text(title, style: TextStyle(fontSize: 19, fontWeight: FontWeight.w600)),
              ],
            ),
            SizedBox(height: 10),
            for (final item in items)
              Padding(
                padding: EdgeInsets.only(bottom: 10),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Padding(
                      padding: EdgeInsets.only(top: 9, right: 12, left: 4),
                      child: CircleAvatar(radius: 2.5, backgroundColor: c.muted),
                    ),
                    Expanded(child: Text(item, style: TextStyle(fontSize: 16, height: 1.5))),
                  ],
                ),
              ),
          ],
          SizedBox(height: 16),
          Text(
            'Kaynak: AFAD vatandaş rehberleri (Deprem Anında Yapılması Gerekenler, İlk 72 Saat).',
            style: TextStyle(color: c.muted, fontSize: 12),
          ),
        ],
      ),
    );
  }
}
