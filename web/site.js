const byId = (id) => document.getElementById(id);

function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  const content = await response.json();
  if (!response.ok) {
    const detail = typeof content.detail === "string" ? content.detail : "请检查填写的内容。";
    throw new Error(detail);
  }
  return content;
}

function status(id, message, error = false) {
  const node = byId(id);
  node.textContent = message;
  node.classList.toggle("error", error);
}

function setIndex(info) {
  byId("index-status").textContent = `${info.documents} 篇资料 · ${info.chunks} 个片段 · ${info.vector_ready ? "向量就绪" : info.vector_status}`;
  const mode = byId("mode");
  for (const option of mode.options) option.disabled = !info.vector_ready && option.value !== "bm25";
  if (!info.vector_ready) mode.value = "bm25";
  const scope = byId("document");
  const selected = scope.value;
  scope.replaceChildren(new Option("全部资料", ""),
    ...info.document_names.map(name => new Option(name, name)));
  if (info.document_names.includes(selected)) scope.value = selected;
}


function renderAnswer(result) {
  const node = byId("answer");
  const types = { hybrid: "混合检索", bm25: "BM25", vector: "向量检索" };
  const answer = element("p", "", "answer-text");
  for (const part of result.answer.split(/(\[\d+\])/u)) {
    const number = /^\[(\d+)\]$/u.exec(part);
    if (number && result.citations.includes(Number(number[1]))) {
      answer.append(citationLink(Number(number[1])));
    } else {
      answer.append(document.createTextNode(part));
    }
  }
  node.replaceChildren(
    element("h2", "回答与依据"),
    element("span", `${types[result.mode]} · ${result.answer_type === "model" ? "模型回答" : "原文摘录"}`, "tag"),
    answer,
  );
  if (result.citations.length && result.answer_type === "model") {
    const citations = element("p", "引用 ", "citations");
    for (const id of result.citations) citations.append(citationLink(id), " ");
    node.append(citations);
  }
  if (result.sources.length) {
    node.append(element("h3", "检索到的原文", "source-heading"));
  }
  for (const source of result.sources) {
    const details = element("details", "");
    details.id = `source-${source.id}`;
    details.open = result.citations.includes(source.id) && source.id === result.citations[0];
    const position = source.heading || (source.page ? `第 ${source.page} 页` : "");
    details.append(
      element("summary", `[${source.id}] ${source.document}${position ? " · " + position : ""}`),
      element("p", source.text, "quote"),
    );
    node.append(details);
  }
}

function citationLink(id) {
  const link = element("a", `[${id}]`, "citation");
  link.href = `#source-${id}`;
  link.setAttribute("aria-label", `打开引用 ${id}`);
  link.addEventListener("click", event => {
    const source = byId(`source-${id}`);
    if (!source) return;
    event.preventDefault();
    source.open = true;
    source.querySelector("summary").focus();
    source.scrollIntoView({ block: "nearest" });
  });
  return link;
}


function handleForm(formId, messageId, loading, submit) {
  const form = byId(formId);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector('[type="submit"]');
    if (button.disabled) return;
    button.disabled = true;
    status(messageId, loading);
    try {
      await submit();
      status(messageId, "");
    } catch (error) {
      status(messageId, error.message, true);
    } finally {
      button.disabled = false;
    }
  });
}


for (const button of document.querySelectorAll("[data-question]")) {
  button.addEventListener("click", () => {
    byId("question").value = button.dataset.question;
    byId("question").focus();
  });
}
handleForm("ask-form", "ask-status", "正在查询资料…", async () => {
  renderAnswer(await request("/api/knowledge/ask", {
    question: byId("question").value, mode: byId("mode").value,
    use_model: byId("answer-mode").value === "model", document: byId("document").value || null,
  }));
});

byId("reindex").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  status("ask-status", "正在读取文档并重建索引…");
  try {
    const info = await request("/api/knowledge/reindex", {});
    setIndex(info);
    const metrics = info.vector_ready
      ? `新增编码 ${info.encoded_passages} 段，复用 ${info.reused_embeddings} 段。` : "";
    status("ask-status", "已更新索引。" + metrics);
  } catch (error) {
    status("ask-status", error.message, true);
  } finally {
    button.disabled = false;
  }
});
request("/api/status").then((info) => {
  setIndex(info);
  for (const id of ["answer-mode"]) {
    const option = byId(id).querySelector('[value="model"]');
    option.disabled = !info.model_ready;
    if (!info.model_ready) option.textContent += "（需配置）";
  }
}).catch((error) => {
  status("index-status", "无法读取索引状态");
  status("ask-status", error.message, true);
});
