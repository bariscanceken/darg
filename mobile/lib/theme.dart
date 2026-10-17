import 'package:flutter/material.dart';

/// Warm palette in a dark and a light version. The accent is the one place to change once the logo arrives.
/// Widgets read the active one with `context.colors`.
@immutable
class DargPalette extends ThemeExtension<DargPalette> {
  const DargPalette({
    required this.background,
    required this.surface,
    required this.surfaceHigh,
    required this.border,
    required this.text,
    required this.muted,
    required this.accent,
    required this.emergency,
    required this.afad,
    required this.web,
    required this.general,
  });

  final Color background;
  final Color surface;
  final Color surfaceHigh;
  final Color border;
  final Color text;
  final Color muted;
  final Color accent;
  final Color emergency;
  final Color afad;
  final Color web;
  final Color general;

  static const dark = DargPalette(
    background: Color(0xFF1F1E1C),
    surface: Color(0xFF2A2927),
    surfaceHigh: Color(0xFF353431),
    border: Color(0xFF3E3D39),
    text: Color(0xFFECE9E2),
    muted: Color(0xFFA19E96),
    accent: Color(0xFFD9774F),
    emergency: Color(0xFFE5484D),
    afad: Color(0xFF6FBF8B),
    web: Color(0xFF6EA8E8),
    general: Color(0xFF8C8981),
  );

  static const light = DargPalette(
    background: Color(0xFFFAF8F4),
    surface: Color(0xFFF1EEE7),
    surfaceHigh: Color(0xFFE7E2D8),
    border: Color(0xFFDCD6CA),
    text: Color(0xFF2A2925),
    muted: Color(0xFF75716A),
    accent: Color(0xFFC2633C),
    emergency: Color(0xFFD42F35),
    afad: Color(0xFF2D8A57),
    web: Color(0xFF2E6DBD),
    general: Color(0xFF85817A),
  );

  @override
  DargPalette copyWith() => this;

  @override
  DargPalette lerp(ThemeExtension<DargPalette>? other, double t) {
    if (other is! DargPalette) return this;
    Color mix(Color a, Color b) => Color.lerp(a, b, t)!;
    return DargPalette(
      background: mix(background, other.background),
      surface: mix(surface, other.surface),
      surfaceHigh: mix(surfaceHigh, other.surfaceHigh),
      border: mix(border, other.border),
      text: mix(text, other.text),
      muted: mix(muted, other.muted),
      accent: mix(accent, other.accent),
      emergency: mix(emergency, other.emergency),
      afad: mix(afad, other.afad),
      web: mix(web, other.web),
      general: mix(general, other.general),
    );
  }
}

extension DargColorsOf on BuildContext {
  DargPalette get colors => Theme.of(this).extension<DargPalette>()!;
}

ThemeData buildTheme(Brightness brightness) {
  final c = brightness == Brightness.dark ? DargPalette.dark : DargPalette.light;
  final base = ThemeData(
    useMaterial3: true,
    brightness: brightness,
    colorScheme: ColorScheme(
      brightness: brightness,
      primary: c.accent,
      onPrimary: Colors.white,
      secondary: c.accent,
      onSecondary: Colors.white,
      surface: c.background,
      onSurface: c.text,
      error: c.emergency,
      onError: Colors.white,
    ),
    scaffoldBackgroundColor: c.background,
    extensions: [c],
  );
  return base.copyWith(
    appBarTheme: AppBarTheme(
      backgroundColor: c.background,
      foregroundColor: c.text,
      elevation: 0,
      scrolledUnderElevation: 0,
      centerTitle: true,
    ),
    drawerTheme: DrawerThemeData(backgroundColor: c.surface),
    dividerColor: c.border,
    dividerTheme: DividerThemeData(color: c.border),
    textTheme: base.textTheme.apply(bodyColor: c.text, displayColor: c.text),
    snackBarTheme: SnackBarThemeData(
      backgroundColor: c.surfaceHigh,
      contentTextStyle: TextStyle(color: c.text),
      behavior: SnackBarBehavior.floating,
    ),
    dialogTheme: DialogThemeData(backgroundColor: c.surface),
    bottomSheetTheme: BottomSheetThemeData(backgroundColor: c.surface),
  );
}
