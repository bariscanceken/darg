// Theme button: cycles light / dark and remembers the choice in this browser. Without a saved choice the system
// setting applies (style.css).
(function () {
  var root = document.documentElement;
  var button = document.getElementById("theme-btn");

  function isDark() {
    if (root.dataset.theme) return root.dataset.theme === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  button.addEventListener("click", function () {
    var next = isDark() ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("darg-theme", next); } catch (e) {}
  });

  document.getElementById("year").textContent = new Date().getFullYear();
})();

// Live demo: asks the DARG API (src/api/app.py, POST /soru and /geri-bildirim) and shows the answer sentence by
// sentence with its sources, like the mobile app. Opened from the computer (localhost or a file), it talks to a local
// API on port 8000. Answer text is always inserted as text, never as HTML.
(function () {
  var local = location.protocol === "file:" || location.hostname === "localhost" || location.hostname === "127.0.0.1";
  var API = local ? "http://localhost:8000" : "https://api.darg.network";
  var STEPS = ["Soru anlaşılıyor", "AFAD belgeleri taranıyor", "Cevap yazılıyor", "Alıntılar doğrulanıyor"];

  var log = document.getElementById("demo-log");
  var empty = document.getElementById("demo-empty");
  var form = document.getElementById("demo-form");
  var input = document.getElementById("demo-input");
  var count = document.getElementById("demo-count");
  var send = document.getElementById("demo-send");
  var newChat = document.getElementById("demo-new");
  var sessionId = null;
  var busy = false;

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function add(node) {
    log.appendChild(node);
    log.scrollTop = log.scrollHeight;
    return node;
  }

  function sourceNumber(label) {  // "Kaynak 3" -> 3
    return parseInt(String(label).replace(/[^0-9]/g, ""), 10) || 0;
  }

  function tagFor(source, text) {
    var kind = source ? { web: "web", il_karti: "gen" }[source.tur] || "afad" : "gen";
    return el("span", "tag " + kind, text);
  }

  function sourceItem(source) {
    var item = el("li");
    item.appendChild(tagFor(source, String(sourceNumber(source.etiket))));
    if (source.tur === "web" && /^https?:\/\//.test(source.kaynak)) {
      var link = el("a", null, source.kaynak.replace(/^https?:\/\/(www\.)?/, ""));
      link.href = source.kaynak;
      link.target = "_blank";
      link.rel = "noopener";
      item.appendChild(link);
      item.appendChild(document.createTextNode(" · resmî web sitesi, bilgi tarihini kontrol edin"));
    } else if (source.tur === "il_karti") {
      item.appendChild(document.createTextNode("İl bilgi kartı (AFAD il afet planından derlendi)"));
    } else {
      var title = source.baslik || source.kaynak;
      if (source.belge_id) {
        var doc = el("a", null, title);
        doc.href = API + "/kutuphane/" + encodeURIComponent(source.belge_id) + "/dosya" +
          (source.ilk_sayfa ? "#page=" + source.ilk_sayfa : "");
        doc.target = "_blank";
        doc.rel = "noopener";
        item.appendChild(doc);
      } else {
        item.appendChild(document.createTextNode(title));
      }
      if (source.sayfa) item.appendChild(document.createTextNode(" · " + source.sayfa));
    }
    return item;
  }

  function feedback(answer, box) {
    var bar = el("div", "rate");
    bar.appendChild(el("span", null, "Cevap işinize yaradı mı?"));
    var good = el("button", null, "👍");
    var bad = el("button", null, "👎");
    good.type = bad.type = "button";
    good.setAttribute("aria-label", "İyi cevap");
    bad.setAttribute("aria-label", "Kötü cevap");
    bar.appendChild(good);
    bar.appendChild(bad);
    var status = bar.appendChild(el("span"));

    function rate(verdict, note) {
      good.setAttribute("aria-pressed", String(verdict === "iyi"));
      bad.setAttribute("aria-pressed", String(verdict === "kotu"));
      var body = { kayit_id: answer.kayit_id, puan: verdict };
      if (note) body.not = note;
      fetch(API + "/geri-bildirim", { method: "POST", headers: { "Content-Type": "application/json" },
                                      body: JSON.stringify(body) })
        .then(function (r) { status.textContent = r.ok ? "Teşekkürler!" : "Kaydedilemedi."; })
        .catch(function () { status.textContent = "Kaydedilemedi."; });
    }

    good.addEventListener("click", function () { rate("iyi"); });
    bad.addEventListener("click", function () {
      rate("kotu");
      if (bar.querySelector(".rate-note")) return;
      var noteRow = el("div", "rate-note");
      var note = el("input");
      note.maxLength = 500;
      note.placeholder = "Ne yanlış ya da eksikti? (isteğe bağlı)";
      var submit = el("button", null, "Gönder");
      submit.type = "button";
      submit.addEventListener("click", function () {
        if (!note.value.trim()) return;
        rate("kotu", note.value.trim());
        noteRow.remove();
      });
      noteRow.appendChild(note);
      noteRow.appendChild(submit);
      bar.appendChild(noteRow);
      note.focus();
    });
    box.appendChild(bar);
  }

  function showAnswer(answer, box) {
    box.textContent = "";
    box.classList.remove("thinking");
    var byNumber = {};
    answer.kaynaklar.forEach(function (s) { byNumber[sourceNumber(s.etiket)] = s; });
    var cited = {};
    if (answer.cumleler.length) {
      answer.cumleler.forEach(function (sentence) {
        var p = el("p", null, sentence.metin);
        if (!sentence.kaynaklar.length) p.appendChild(tagFor(null, "genel bilgi"));
        sentence.kaynaklar.forEach(function (n) {
          cited[n] = true;
          p.appendChild(tagFor(byNumber[n], String(n)));
        });
        box.appendChild(p);
      });
    } else {
      box.appendChild(el("p", null, answer.cevap));
    }
    var list = el("ul", "sources");
    answer.kaynaklar
      .filter(function (s) { return cited[sourceNumber(s.etiket)]; })
      .sort(function (a, b) { return sourceNumber(a.etiket) - sourceNumber(b.etiket); })
      .forEach(function (s) { list.appendChild(sourceItem(s)); });
    if (list.children.length) box.appendChild(list);
    if (answer.cumleler.some(function (s) { return !s.kaynaklar.length; })) {
      box.appendChild(el("p", "muted", "\"genel bilgi\": AFAD belgelerinde doğrulanmamış cümle."));
    }
    feedback(answer, box);
  }

  function showError(message, box) {
    box.textContent = "";
    box.classList.remove("thinking");
    box.classList.add("err");
    box.appendChild(el("p", null, message));
  }

  function errorMessage(status, body) {
    if (body && typeof body.detail === "string") return body.detail;  // the API's own message (limits, outage)
    if (status === 422) return "Soru 1 ile 500 karakter arasında olmalı.";
    return "Bir sorun oluştu (" + status + "). Lütfen biraz sonra tekrar deneyin. Acil bir durumdaysanız 112'yi arayın.";
  }

  function setBusy(value) {
    busy = value;
    send.disabled = value;
    send.textContent = value ? "Bekleyin…" : "Sor";
  }

  function ask(text) {
    text = text.trim();
    if (!text || busy) return;
    empty.hidden = true;
    newChat.hidden = false;
    add(el("div", "bubble q", text));
    var box = add(el("div", "bubble a thinking", STEPS[0]));
    var step = 0;
    var timer = setInterval(function () {
      step = Math.min(step + 1, STEPS.length - 1);
      box.textContent = STEPS[step];
    }, 4000);
    input.value = "";
    updateCount();
    setBusy(true);

    var body = { soru: text };
    if (sessionId) body.oturum_id = sessionId;
    fetch(API + "/soru", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(function (response) {
        return response.json().catch(function () { return null; }).then(function (data) {
          if (!response.ok) throw new Error(errorMessage(response.status, data));
          return data;
        });
      })
      .then(function (answer) {
        sessionId = answer.oturum_id;
        showAnswer(answer, box);
      })
      .catch(function (error) {
        var offline = error instanceof TypeError;  // fetch could not reach the server at all
        showError(offline ? "Sunucuya ulaşılamadı. İnternet bağlantınızı kontrol edip tekrar deneyin. " +
                            "Acil bir durumdaysanız 112'yi arayın." : error.message, box);
      })
      .then(function () {
        clearInterval(timer);
        setBusy(false);
        log.scrollTop = log.scrollHeight;
      });
  }

  function updateCount() { count.textContent = input.value.length + "/500"; }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    ask(input.value);
  });
  input.addEventListener("input", updateCount);
  input.addEventListener("keydown", function (event) {  // Enter sends, Shift+Enter starts a new line
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ask(input.value);
    }
  });
  log.querySelectorAll(".chip").forEach(function (chip) {
    chip.addEventListener("click", function () { ask(chip.textContent); });
  });
  newChat.addEventListener("click", function () {
    if (busy) return;
    sessionId = null;
    Array.prototype.slice.call(log.children).forEach(function (child) { if (child !== empty) child.remove(); });
    empty.hidden = false;
    newChat.hidden = true;
  });
})();
