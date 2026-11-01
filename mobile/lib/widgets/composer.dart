import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme.dart';

class Composer extends StatefulWidget {
  const Composer({super.key, required this.onSend, required this.enabled});

  final void Function(String text) onSend;
  final bool enabled;

  @override
  State<Composer> createState() => _ComposerState();
}

class _ComposerState extends State<Composer> {
  final _controller = TextEditingController();
  final _focus = FocusNode();

  @override
  void initState() {
    super.initState();
    _controller.addListener(() => setState(() {}));
  }

  @override
  void dispose() {
    _controller.dispose();
    _focus.dispose();
    super.dispose();
  }

  bool get _canSend => widget.enabled && _controller.text.trim().isNotEmpty;

  void _send() {
    if (!_canSend) return;
    widget.onSend(_controller.text.trim());
    _controller.clear();
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return SafeArea(
      top: false,
      child: Padding(
        padding: EdgeInsets.fromLTRB(12, 6, 12, 10),
        child: Container(
          decoration: BoxDecoration(
            color: c.surface,
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: c.border),
          ),
          padding: EdgeInsets.fromLTRB(18, 4, 6, 4),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                child: CallbackShortcuts(
                  // Enter sends on web/desktop; Shift+Enter adds a new line.
                  bindings: {SingleActivator(LogicalKeyboardKey.enter): _send},
                  child: TextField(
                    controller: _controller,
                    focusNode: _focus,
                    minLines: 1,
                    maxLines: 6,
                    maxLength: 500, // the server accepts questions up to 500 characters
                    textCapitalization: TextCapitalization.sentences,
                    style: TextStyle(fontSize: 16, color: c.text),
                    cursorColor: c.accent,
                    decoration: InputDecoration(
                      hintText: 'Depremle ilgili sorunu yaz…',
                      hintStyle: TextStyle(color: c.muted),
                      border: InputBorder.none,
                      counterText: '',
                      contentPadding: EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
                ),
              ),
              Padding(
                padding: EdgeInsets.only(bottom: 4),
                child: AnimatedContainer(
                  duration: Duration(milliseconds: 150),
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(color: _canSend ? c.accent : c.surfaceHigh, shape: BoxShape.circle),
                  child: IconButton(
                    onPressed: _canSend ? _send : null,
                    tooltip: 'Gönder',
                    icon: Icon(Icons.arrow_upward_rounded, color: _canSend ? Colors.white : c.muted, size: 22),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
