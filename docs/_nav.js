// Shared sidebar nav data
const NAV = [
  { label: "Getting Started", items: [
    { href: "index.html",       text: "Introduction" },
    { href: "quickstart.html",  text: "Quick Start" },
  ]},
  { label: "Core Concepts", items: [
    { href: "architecture.html",  text: "Architecture" },
    { href: "platforms.html",     text: "Platforms" },
    { href: "robot-loading.html", text: "Robot Loading" },
  ]},
  { label: "Reference", items: [
    { href: "api-reference.html", text: "API Reference" },
    { href: "plugin-sdk.html",    text: "Plugin SDK" },
  ]},
  { label: "Operations", items: [
    { href: "safety.html",      text: "Safety" },
    { href: "deployment.html",  text: "Deployment" },
  ]},
];

// Search index
const PAGES = [
  { title: "Introduction",    href: "index.html",        keywords: "physicore mpc overview features" },
  { title: "Quick Start",     href: "quickstart.html",   keywords: "install pip clone serve dashboard" },
  { title: "Architecture",    href: "architecture.html", keywords: "cem optimizer state estimator fleet ekf residual" },
  { title: "Platforms",       href: "platforms.html",    keywords: "quadrotor balancing_bot legged satellite dynamics" },
  { title: "Robot Loading",   href: "robot-loading.html",keywords: "urdf yaml load_robot build_robot_model joints" },
  { title: "API Reference",   href: "api-reference.html",keywords: "rest endpoints fastapi health fleet telemetry" },
  { title: "Plugin SDK",      href: "plugin-sdk.html",   keywords: "plugin on_step on_load marketplace safety scan" },
  { title: "Safety",          href: "safety.html",       keywords: "sentinel kill switch e-stop bounds watchdog" },
  { title: "Deployment",      href: "deployment.html",   keywords: "docker compose kubernetes cloud production" },
];

function buildNav(currentPage) {
  const sidebar = document.getElementById("sidebar");
  const nav = document.createElement("nav");

  NAV.forEach(section => {
    const lbl = document.createElement("div");
    lbl.className = "section-label";
    lbl.textContent = section.label;
    nav.appendChild(lbl);

    section.items.forEach(item => {
      const a = document.createElement("a");
      a.href = item.href;
      a.textContent = item.text;
      if (item.href === currentPage) a.className = "active";
      nav.appendChild(a);
    });
  });

  sidebar.appendChild(nav);
}

function buildSearch() {
  const wrap = document.getElementById("search-wrap");
  const input = document.getElementById("search-input");
  const results = document.getElementById("search-results");

  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    results.innerHTML = "";
    if (!q) { results.style.display = "none"; return; }

    const hits = PAGES.filter(p =>
      p.title.toLowerCase().includes(q) || p.keywords.includes(q)
    );
    if (!hits.length) { results.style.display = "none"; return; }

    hits.forEach(p => {
      const a = document.createElement("a");
      a.href = p.href;
      a.textContent = p.title;
      results.appendChild(a);
    });
    results.style.display = "block";
  });

  document.addEventListener("click", e => {
    if (!wrap.contains(e.target)) results.style.display = "none";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  buildNav(window.location.pathname.split("/").pop() || "index.html");
  buildSearch();
});
