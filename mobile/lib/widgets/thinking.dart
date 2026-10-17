import 'dart:async';

import 'package:flutter/material.dart';

import '../theme.dart';

/// Shown while an answer is being prepared (~10 s, longer when the web is searched).
/// The API answers in one response, so the steps follow typical timings, not live progress.
class ThinkingIndicator extends StatefulWidget {
  const ThinkingIndicator({super.key});

  @override
  State<ThinkingIndicator> createState() => _ThinkingIndicatorState();
}

class _ThinkingIndicatorState extends State<ThinkingIndicator> with SingleTickerProviderStateMixin {
  static const _steps = [
    (0, 'Soru anlaşılıyor'),
    (2, 'AFAD belgeleri aranıyor'),
    (5, 'Cevap yazılıyor'),
    (14, 'Resmî kaynaklar kontrol ediliyor'),
    (35, 'Biraz daha sürüyor'),
  ];

  late final AnimationController _pulse = AnimationController(vsync: this, duration: Duration(milliseconds: 900))
    ..repeat(reverse: true);
  late final Timer _timer;
  final _clock = Stopwatch()..start();

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(Duration(milliseconds: 500), (_) => setState(() {}));
  }

  @override
  void dispose() {
    _timer.cancel();
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final seconds = _clock.elapsed.inSeconds;
    final label = _steps.lastWhere((step) => seconds >= step.$1).$2;
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 14),
      child: Row(
        children: [
          FadeTransition(
            opacity: Tween(begin: 0.35, end: 1.0).animate(_pulse),
            child: Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(color: c.accent, shape: BoxShape.circle),
            ),
          ),
          SizedBox(width: 12),
          AnimatedSwitcher(
            duration: Duration(milliseconds: 250),
            child: Text(
              '$label…',
              key: ValueKey(label),
              style: TextStyle(color: c.muted, fontSize: 15),
            ),
          ),
        ],
      ),
    );
  }
}
