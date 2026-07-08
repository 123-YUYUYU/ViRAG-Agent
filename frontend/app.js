const tabs = document.querySelectorAll(".query-tab");
const panels = document.querySelectorAll(".query-panel");
const modeChips = document.querySelectorAll(".mode-chip");
const navItems = document.querySelectorAll(".nav-item");
const submitButton = document.querySelector("#submit-query");
const queryInput = document.querySelector("#manual-query");
const chatStack = document.querySelector("#chat-stack");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;

    tabs.forEach((item) => {
      const isActive = item === tab;
      item.classList.toggle("is-active", isActive);
      item.setAttribute("aria-selected", String(isActive));
    });

    panels.forEach((panel) => {
      panel.classList.toggle("is-active", panel.dataset.panel === target);
    });
  });
});

modeChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    chip.classList.toggle("is-on");
  });
});

navItems.forEach((item) => {
  item.addEventListener("click", () => {
    navItems.forEach((nav) => nav.classList.remove("is-active"));
    item.classList.add("is-active");
  });
});

submitButton.addEventListener("click", () => {
  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }

  const userMessage = document.createElement("div");
  userMessage.className = "message user-message";
  userMessage.innerHTML = `<span>${icon("user")}User</span><p>${escapeHtml(query)}</p>`;

  const assistantMessage = document.createElement("div");
  assistantMessage.className = "message assistant-message";
  assistantMessage.innerHTML = `
    <span>${icon("bot")}ViRAG-Agent</span>
    <p>
      Dynamic routing has been simulated: the short query is first expanded under semantic constraints,
      then CLIP and BM25 perform multi-route recall. RRF fusion and BGE reranking select the evidence page,
      and Qwen3-VL reads the original page image before the answer is written back into session memory.
    </p>
    <div class="source-row">
      <mark>${icon("search")}Mock Retrieval</mark>
      <mark>${icon("evidence")}Evidence Page</mark>
      <mark>${icon("memory")}Memory Updated</mark>
    </div>
  `;

  chatStack.append(userMessage, assistantMessage);
  assistantMessage.scrollIntoView({ behavior: "smooth", block: "nearest" });
});

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function icon(name) {
  return `<svg class="ui-icon" aria-hidden="true"><use href="#icon-${name}"></use></svg>`;
}
