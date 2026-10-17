import 'dart:io';

import 'package:flutter/material.dart';
import 'package:pdfrx/pdfrx.dart';

import '../library.dart';
import '../theme.dart';

/// Opens a library document: the downloaded copy if there is one (works offline), otherwise from the server.
class DocumentViewerScreen extends StatelessWidget {
  const DocumentViewerScreen({super.key, required this.doc, this.page = 1});

  final LibraryDoc doc;
  final int page;

  /// Opens a document known only by id (e.g. an answer's source); looks it up in the downloads or the catalog.
  static Future<void> openById(BuildContext context, String id, {String? title, int? page}) async {
    final store = LibraryStore.instance;
    var doc = store.downloads[id];
    if (doc == null) {
      final catalog = await store.loadCatalog();
      doc = catalog.documents.where((d) => d.id == id).firstOrNull;
    }
    if (!context.mounted) return;
    if (doc == null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Belge açılamadı. İnternet bağlantınızı kontrol edin.')));
      return;
    }
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => DocumentViewerScreen(doc: doc!, page: page ?? 1),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    final store = LibraryStore.instance;
    return ListenableBuilder(
      listenable: store,
      builder: (context, _) {
        final local = store.isDownloaded(doc.id) ? store.localFile(doc) : null;
        final downloading = store.progress[doc.id];
        return Scaffold(
          appBar: AppBar(
            centerTitle: false,
            title: Text(doc.title, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 17)),
            actions: [
              if (store.canDownload && local == null)
                downloading != null
                    ? Padding(
                        padding: const EdgeInsets.all(14),
                        child: SizedBox(
                          width: 22,
                          height: 22,
                          child: CircularProgressIndicator(value: downloading, strokeWidth: 2.5),
                        ),
                      )
                    : IconButton(
                        tooltip: 'İndir (${doc.sizeLabel})',
                        icon: const Icon(Icons.download_outlined),
                        onPressed: () => store.download(doc).catchError((_) {
                          if (context.mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('İndirilemedi')));
                          }
                        }),
                      ),
              if (local != null)
                Padding(
                  padding: const EdgeInsets.only(right: 12),
                  child: Icon(Icons.offline_pin_rounded, color: c.afad),
                ),
            ],
          ),
          body: doc.isImage ? _image(local) : _pdf(local),
        );
      },
    );
  }

  Widget _pdf(File? local) {
    final params = PdfViewerParams(
      backgroundColor: Colors.grey.shade800,
      errorBannerBuilder: (context, error, stackTrace, documentRef) => const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'Belge açılamadı. İnternet bağlantınızı kontrol edin ya da belgeyi indirip tekrar deneyin.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white),
          ),
        ),
      ),
    );
    if (local != null) return PdfViewer.file(local.path, initialPageNumber: page, params: params);
    return PdfViewer.uri(LibraryStore.instance.fileUri(doc.id), initialPageNumber: page, params: params);
  }

  Widget _image(File? local) {
    final image = local != null
        ? Image.file(local)
        : Image.network(
            LibraryStore.instance.fileUri(doc.id).toString(),
            errorBuilder: (_, _, _) => const Text('Görsel açılamadı. İnternet bağlantınızı kontrol edin.'),
          );
    return InteractiveViewer(maxScale: 6, child: Center(child: image));
  }
}
