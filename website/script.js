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
