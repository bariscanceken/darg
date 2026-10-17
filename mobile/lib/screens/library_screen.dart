import 'dart:async';

import 'package:flutter/material.dart';

import '../library.dart';
import '../theme.dart';
import 'document_viewer_screen.dart';

/// All AFAD source documents: read online, download for offline use, search the downloads without internet.
class LibraryScreen extends StatefulWidget {
  const LibraryScreen({super.key});

  @override
  State<LibraryScreen> createState() => _LibraryScreenState();
}

class _LibraryScreenState extends State<LibraryScreen> {
  final _store = LibraryStore.instance;
  LibraryCatalog? _catalog;
  String _filter = '';

  @override
  void initState() {
    super.initState();
    _reload();
  }

  Future<void> _reload() async {
    final catalog = await _store.loadCatalog();
    if (mounted) setState(() => _catalog = catalog);
  }

  void _open(LibraryDoc doc, {int page = 1}) {
    if (_catalog!.offline && !_store.isDownloaded(doc.id)) {
      _snack('İnternet yok: bu belge indirilmediği için açılamıyor.');
      return;
    }
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => DocumentViewerScreen(doc: doc, page: page),
      ),
    );
  }

  void _snack(String text) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));

  Future<void> _download(LibraryDoc doc) async {
    if (doc.size >= LibraryStore.largeFileBytes) {
      final ok = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Büyük dosya'),
          content: Text(
            '"${doc.title}" ${doc.sizeLabel}. Mobil veriyle indirmek kotanızı harcayabilir. İndirilsin mi?',
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Vazgeç')),
            FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('İndir')),
          ],
        ),
      );
      if (ok != true) return;
    }
    try {
      await _store.download(doc);
    } catch (_) {
      if (mounted) _snack('"${doc.title}" indirilemedi. İnternet bağlantınızı kontrol edin.');
    }
  }

  Future<void> _downloadCore(List<LibraryDoc> docs) async {
    var failed = 0;
    for (final doc in docs.where((d) => !_store.isDownloaded(d.id))) {
      try {
        await _store.download(doc);
      } catch (_) {
        failed++;
      }
    }
    if (mounted && failed > 0) _snack('$failed belge indirilemedi. Tekrar deneyebilirsiniz.');
  }

  Future<void> _remove(LibraryDoc doc) async {
    await _store.remove(doc);
    if (mounted) _snack('"${doc.title}" telefondan silindi');
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Kütüphane'),
          bottom: const TabBar(
            tabs: [
              Tab(text: 'Belgeler'),
              Tab(text: 'İndirilenler'),
            ],
          ),
        ),
        body: _catalog == null
            ? const Center(child: CircularProgressIndicator())
            : ListenableBuilder(
                listenable: _store,
                builder: (context, _) => TabBarView(
                  children: [
                    _documentsTab(),
                    _DownloadsTab(store: _store, onOpen: _open, onRemove: _remove),
                  ],
                ),
              ),
      ),
    );
  }

  Widget _documentsTab() {
    final c = context.colors;
    final catalog = _catalog!;
    final query = foldTurkish(_filter.trim());
    final visible = catalog.documents
        .where((d) => query.isEmpty || foldTurkish('${d.title} ${d.province ?? ''} ${d.categoryName}').contains(query))
        .toList();
    final categories = <String, List<LibraryDoc>>{};
    for (final doc in visible) {
      categories.putIfAbsent(doc.categoryName, () => []).add(doc);
    }
    return RefreshIndicator(
      onRefresh: _reload,
      child: ListView(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
        children: [
          if (catalog.offline)
            Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: c.surface,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: c.border),
              ),
              child: Row(
                children: [
                  Icon(Icons.wifi_off_rounded, color: c.muted),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'İnternet yok: yalnız indirdiğiniz belgeler açılabilir. Asistan da internetsiz çalışmaz.',
                      style: TextStyle(color: c.text, height: 1.4),
                    ),
                  ),
                ],
              ),
            ),
          if (_store.canDownload && query.isEmpty && catalog.core.isNotEmpty) _corePackage(catalog.core),
          TextField(
            onChanged: (value) => setState(() => _filter = value),
            decoration: InputDecoration(
              hintText: 'Belge ya da il ara',
              prefixIcon: const Icon(Icons.search),
              isDense: true,
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
            ),
          ),
          const SizedBox(height: 8),
          if (visible.isEmpty)
            Padding(
              padding: const EdgeInsets.all(24),
              child: Text(
                'Sonuç yok',
                textAlign: TextAlign.center,
                style: TextStyle(color: c.muted),
              ),
            ),
          for (final entry in categories.entries)
            Theme(
              data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
              child: ExpansionTile(
                key: PageStorageKey('${entry.key}-${query.isNotEmpty}'),
                initiallyExpanded: query.isNotEmpty || entry.value.first.category == 'vatandas_rehberi',
                tilePadding: const EdgeInsets.symmetric(horizontal: 4),
                title: Text(entry.key, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                subtitle: Text('${entry.value.length} belge', style: TextStyle(color: c.muted, fontSize: 12.5)),
                children: [for (final doc in entry.value) _docTile(doc)],
              ),
            ),
        ],
      ),
    );
  }

  Widget _corePackage(List<LibraryDoc> core) {
    final c = context.colors;
    final done = core.where((d) => _store.isDownloaded(d.id)).length;
    final busy = core.any((d) => _store.progress.containsKey(d.id));
    final total = core.fold<int>(0, (sum, d) => sum + d.size);
    final complete = done == core.length;
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: complete ? c.afad : c.accent.withValues(alpha: 0.6)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                complete ? Icons.offline_pin_rounded : Icons.library_books_outlined,
                color: complete ? c.afad : c.accent,
              ),
              const SizedBox(width: 10),
              const Expanded(
                child: Text('Temel rehberler paketi', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600)),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            '${core.length} belge · ${formatBytes(total)} · İlk 72 Saat, Psikolojik İlk Yardım, Toplanma Alanları, '
            'Afet Çantası ve diğer vatandaş rehberleri. İnternet yokken de okunabilir.',
            style: TextStyle(color: c.muted, fontSize: 13.5, height: 1.4),
          ),
          const SizedBox(height: 12),
          if (busy) ...[
            LinearProgressIndicator(value: done / core.length, minHeight: 6, borderRadius: BorderRadius.circular(3)),
            const SizedBox(height: 6),
            Text('İndiriliyor: $done / ${core.length}', style: TextStyle(color: c.muted, fontSize: 13)),
          ] else if (complete)
            Text(
              'Telefonunuzda ✓',
              style: TextStyle(color: c.afad, fontWeight: FontWeight.w600),
            )
          else
            FilledButton.icon(
              onPressed: _catalog!.offline ? null : () => _downloadCore(core),
              icon: const Icon(Icons.download_rounded),
              label: Text(done == 0 ? 'Paketi indir' : 'Kalanları indir (${core.length - done})'),
            ),
        ],
      ),
    );
  }

  Widget _docTile(LibraryDoc doc) {
    final c = context.colors;
    final downloaded = _store.isDownloaded(doc.id);
    final progress = _store.progress[doc.id];
    final details = [
      doc.sizeLabel,
      if (doc.pages != null && !doc.isImage) '${doc.pages} sayfa',
      if (doc.isImage) 'görsel',
    ].join(' · ');
    Widget trailing;
    if (!_store.canDownload) {
      trailing = const SizedBox.shrink();
    } else if (progress != null) {
      trailing = SizedBox(width: 24, height: 24, child: CircularProgressIndicator(value: progress, strokeWidth: 2.5));
    } else if (downloaded) {
      trailing = PopupMenuButton<String>(
        icon: Icon(Icons.offline_pin_rounded, color: c.afad),
        tooltip: 'İndirildi',
        onSelected: (_) => _remove(doc),
        itemBuilder: (_) => [const PopupMenuItem(value: 'sil', child: Text('Telefondan sil'))],
      );
    } else {
      trailing = IconButton(
        tooltip: 'İndir (${doc.sizeLabel})',
        icon: Icon(Icons.download_outlined, color: _catalog!.offline ? c.border : c.muted),
        onPressed: _catalog!.offline ? null : () => _download(doc),
      );
    }
    return ListTile(
      contentPadding: const EdgeInsets.only(left: 4),
      leading: Icon(doc.isImage ? Icons.image_outlined : Icons.picture_as_pdf_outlined, color: c.muted),
      title: Text(doc.title, style: const TextStyle(fontSize: 15)),
      subtitle: Text(details, style: TextStyle(color: c.muted, fontSize: 12.5)),
      trailing: trailing,
      onTap: () => _open(doc),
    );
  }
}

class _DownloadsTab extends StatefulWidget {
  const _DownloadsTab({required this.store, required this.onOpen, required this.onRemove});

  final LibraryStore store;
  final void Function(LibraryDoc doc, {int page}) onOpen;
  final void Function(LibraryDoc doc) onRemove;

  @override
  State<_DownloadsTab> createState() => _DownloadsTabState();
}

class _DownloadsTabState extends State<_DownloadsTab> {
  String _query = '';
  List<SearchHit>? _hits;
  Timer? _debounce;

  @override
  void dispose() {
    _debounce?.cancel();
    super.dispose();
  }

  void _search(String value) {
    _query = value;
    _debounce?.cancel();
    _debounce = Timer(const Duration(milliseconds: 300), () async {
      final hits = value.trim().length < 2 ? null : await widget.store.searchOffline(value);
      if (mounted && value == _query) setState(() => _hits = hits);
    });
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final store = widget.store;
    if (!store.canDownload) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            'İndirme yalnızca telefon uygulamasında kullanılabilir.',
            textAlign: TextAlign.center,
            style: TextStyle(color: c.muted),
          ),
        ),
      );
    }
    final docs = store.downloads.values.toList()..sort((a, b) => a.title.compareTo(b.title));
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
      children: [
        TextField(
          onChanged: _search,
          decoration: InputDecoration(
            hintText: 'İndirilen belgelerde ara (internetsiz)',
            prefixIcon: const Icon(Icons.manage_search_rounded),
            isDense: true,
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
          ),
        ),
        const SizedBox(height: 10),
        if (_hits != null) ...[
          Text(
            _hits!.isEmpty ? 'İndirilen belgelerde bulunamadı' : '${_hits!.length} sayfada bulundu',
            style: TextStyle(color: c.muted, fontSize: 13),
          ),
          for (final hit in _hits!)
            ListTile(
              contentPadding: const EdgeInsets.symmetric(horizontal: 4),
              title: Text(
                '${hit.doc.title} · s. ${hit.page}',
                style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
              ),
              subtitle: Text(hit.snippet, style: TextStyle(color: c.text, fontSize: 13.5, height: 1.4)),
              onTap: () => widget.onOpen(hit.doc, page: hit.page),
            ),
        ] else ...[
          Text(
            docs.isEmpty
                ? 'Henüz belge indirmediniz. "Belgeler" sekmesinden temel rehberler paketini indirebilirsiniz.'
                : '${docs.length} belge · ${formatBytes(store.usedBytes)} kullanılıyor',
            style: TextStyle(color: c.muted, fontSize: 13.5, height: 1.4),
          ),
          const SizedBox(height: 6),
          for (final doc in docs)
            ListTile(
              contentPadding: const EdgeInsets.only(left: 4),
              leading: Icon(Icons.offline_pin_rounded, color: c.afad),
              title: Text(doc.title, style: const TextStyle(fontSize: 15)),
              subtitle: Text(
                doc.searchable ? doc.sizeLabel : '${doc.sizeLabel} · metin araması yok (görsel belge)',
                style: TextStyle(color: c.muted, fontSize: 12.5),
              ),
              trailing: IconButton(
                tooltip: 'Telefondan sil',
                icon: Icon(Icons.delete_outline, color: c.muted),
                onPressed: () => widget.onRemove(doc),
              ),
              onTap: () => widget.onOpen(doc),
            ),
        ],
      ],
    );
  }
}
