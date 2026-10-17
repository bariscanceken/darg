import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_volume_controller/flutter_volume_controller.dart';
import 'package:torch_light/torch_light.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../theme.dart';

/// SOS in Morse (· · · — — — · · ·) as (light on?, duration) steps; one unit = 200 ms.
const _unit = Duration(milliseconds: 200);
const sosPattern = [
  (true, 1), (false, 1), (true, 1), (false, 1), (true, 1), (false, 3), //
  (true, 3), (false, 1), (true, 3), (false, 1), (true, 3), (false, 3), //
  (true, 1), (false, 1), (true, 1), (false, 1), (true, 1), (false, 7),
];

/// Help signals for someone trapped or lost: a loud rescue whistle (three blasts, the distress signal),
/// the flashlight blinking SOS, and the screen blinking SOS (works on every phone).
class SignalScreen extends StatefulWidget {
  const SignalScreen({super.key});

  @override
  State<SignalScreen> createState() => _SignalScreenState();
}

class _SignalScreenState extends State<SignalScreen> {
  final _player = AudioPlayer();
  bool _whistle = false;
  bool _torch = false;
  bool _screen = false;
  bool _screenLit = false;
  double? _volumeBefore;
  int _torchRun = 0;
  int _screenRun = 0;

  bool get _anyOn => _whistle || _torch || _screen;

  static bool get _mobile =>
      !kIsWeb && (defaultTargetPlatform == TargetPlatform.android || defaultTargetPlatform == TargetPlatform.iOS);

  @override
  void dispose() {
    _torchRun++;
    _screenRun++;
    _player.stop();
    _player.dispose();
    if (_torch) TorchLight.disableTorch().catchError((_) {});
    _restoreVolume();
    WakelockPlus.disable().catchError((_) {});
    super.dispose();
  }

  Future<void> _restoreVolume() async {
    if (_volumeBefore != null && _mobile) {
      await FlutterVolumeController.setVolume(_volumeBefore!).catchError((_) {});
      _volumeBefore = null;
    }
  }

  void _keepAwake() => (_anyOn ? WakelockPlus.enable() : WakelockPlus.disable()).catchError((_) {});

  Future<void> _toggleWhistle() async {
    if (_whistle) {
      await _player.stop();
      await _restoreVolume();
    } else {
      if (_mobile) {
        try {
          _volumeBefore = await FlutterVolumeController.getVolume();
          await FlutterVolumeController.setVolume(1.0);
        } catch (_) {}
      }
      await _player.setReleaseMode(ReleaseMode.loop);
      await _player.play(AssetSource('sounds/whistle.wav'), volume: 1.0);
    }
    setState(() => _whistle = !_whistle);
    _keepAwake();
  }

  Future<void> _toggleTorch() async {
    if (_torch) {
      _torchRun++;
      setState(() => _torch = false);
      await TorchLight.disableTorch().catchError((_) {});
      _keepAwake();
      return;
    }
    var available = false;
    try {
      available = _mobile && await TorchLight.isTorchAvailable();
    } catch (_) {}
    if (!available) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Bu cihazda fener kullanılamıyor. Ekran ışığını deneyin.')));
      }
      return;
    }
    setState(() => _torch = true);
    _keepAwake();
    final run = ++_torchRun;
    while (mounted && run == _torchRun) {
      for (final (on, units) in sosPattern) {
        if (!mounted || run != _torchRun) break;
        try {
          await (on ? TorchLight.enableTorch() : TorchLight.disableTorch());
        } catch (_) {}
        await Future<void>.delayed(_unit * units);
      }
    }
  }

  Future<void> _toggleScreen() async {
    if (_screen) {
      _screenRun++;
      setState(() => _screen = _screenLit = false);
      _keepAwake();
      return;
    }
    setState(() => _screen = true);
    _keepAwake();
    final run = ++_screenRun;
    while (mounted && run == _screenRun) {
      for (final (on, units) in sosPattern) {
        if (!mounted || run != _screenRun) break;
        setState(() => _screenLit = on);
        await Future<void>.delayed(_unit * units);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    if (_screen) {
      // Full-screen blinking; a tap anywhere stops it.
      return GestureDetector(
        onTap: _toggleScreen,
        child: Scaffold(
          backgroundColor: _screenLit ? Colors.white : Colors.black,
          body: SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  'SOS · durdurmak için dokunun',
                  style: TextStyle(color: _screenLit ? Colors.black54 : Colors.white54, fontSize: 15),
                ),
              ),
            ),
          ),
        ),
      );
    }
    return Scaffold(
      appBar: AppBar(title: const Text('Yardım sinyali')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(18, 8, 18, 32),
        children: [
          Text(
            'Enkaz altındaysanız ya da kaybolduysanız, kurtarma ekiplerinin sizi bulması için ses ve ışık sinyali '
            'verin. Ekran açık kalır.',
            style: TextStyle(color: c.muted, height: 1.5),
          ),
          const SizedBox(height: 18),
          _SignalButton(
            icon: Icons.campaign_rounded,
            title: 'Düdük',
            subtitle: 'Yüksek sesle üçlü düdük çalar (uluslararası yardım işareti)',
            active: _whistle,
            color: c.emergency,
            onTap: _toggleWhistle,
          ),
          const SizedBox(height: 12),
          _SignalButton(
            icon: Icons.flashlight_on_rounded,
            title: 'SOS feneri',
            subtitle: 'Fener · · · — — — · · · şeklinde yanıp söner',
            active: _torch,
            color: c.accent,
            onTap: _toggleTorch,
          ),
          const SizedBox(height: 12),
          _SignalButton(
            icon: Icons.light_mode_rounded,
            title: 'Ekran ışığı',
            subtitle: 'Ekran SOS şeklinde beyaz yanıp söner; feneri olmayan telefonlarda da çalışır',
            active: false,
            color: c.accent,
            onTap: _toggleScreen,
          ),
          const SizedBox(height: 24),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: c.surface,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: c.border),
            ),
            child: Text(
              'Pilinizi koruyun: sinyali sürekli değil, aralıklarla kullanın; yakınlarda ses duyduğunuzda açın. '
              'Bağırmak yerine düdük kullanın, enerjinizi korursunuz. Telefonunuz çekiyorsa 112\'yi arayın.',
              style: TextStyle(fontSize: 14, height: 1.5, color: c.text),
            ),
          ),
        ],
      ),
    );
  }
}

class _SignalButton extends StatelessWidget {
  const _SignalButton({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.active,
    required this.color,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final bool active;
  final Color color;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Material(
      color: active ? color : c.surface,
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: active ? color : c.border, width: 1.5),
          ),
          child: Row(
            children: [
              Icon(icon, size: 40, color: active ? Colors.white : color),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      active ? '$title · AÇIK' : title,
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                        color: active ? Colors.white : c.text,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      active ? 'Durdurmak için dokunun' : subtitle,
                      style: TextStyle(fontSize: 13.5, height: 1.35, color: active ? Colors.white70 : c.muted),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
