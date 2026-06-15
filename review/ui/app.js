const state = {
  items: [],
  index: 0,
  mode: "review",        // "review" 人工审批 | "compare" 模型对比
  filter: "pending",     // 审批模式下按状态筛选；对比模式固定 "compare"
};

function currentFilter() {
  return state.mode === "compare" ? "compare" : state.filter;
}

async function loadItems() {
  const res = await fetch(`/api/items?filter=${encodeURIComponent(currentFilter())}`);
  const data = await res.json();
  state.items = data.items;
  state.summary = data.summary;
  if (state.index >= state.items.length) state.index = Math.max(0, state.items.length - 1);
  render();
}

function renderSummary() {
  const s = state.summary || { total: 0, reviewed: 0, approved: 0, rejected: 0, with_prediction: 0 };
  if (state.mode === "compare") {
    const models = s.models || [];
    document.getElementById("summary").innerHTML = `
      <p>总样本: <strong>${s.total}</strong></p>
      <p>有预测样本: <strong>${s.with_prediction ?? 0}</strong></p>
      <p>对比模型: <strong>${models.length}</strong></p>
    `;
    const hint = document.getElementById("compareHint");
    hint.textContent = models.length === 0
      ? "未加载模型预测：启动时用 --predictions 标签=路径 传入（可多次）。"
      : `对比模型：${models.join(" / ")}（${s.with_prediction} 条有预测）。`;
  } else {
    document.getElementById("summary").innerHTML = `
      <p>总数: <strong>${s.total}</strong></p>
      <p>已审: <strong>${s.reviewed}</strong></p>
      <p>通过: <strong>${s.approved}</strong></p>
      <p>不通过: <strong>${s.rejected}</strong></p>
    `;
  }
}

function clearDetail() {
  document.getElementById("image").src = "";
  document.getElementById("question").textContent = "";
  document.getElementById("answerReview").textContent = "";
  document.getElementById("compareCols").innerHTML = "";
  document.getElementById("meta").textContent = "";
  document.getElementById("reviewStatus").textContent = "";
  document.getElementById("note").value = "";
}

// 对比模式：按 item.predictions 顺序，每个模型渲染一列
function renderCompareColumns(item) {
  const container = document.getElementById("compareCols");
  container.innerHTML = "";
  const preds = item.predictions || [];
  if (!preds.length) {
    container.innerHTML = `<p class="hint">未加载模型预测：启动时用 --predictions 标签=路径 传入（可多次）。</p>`;
    return;
  }
  for (const p of preds) {
    const col = document.createElement("div");
    col.className = "compareCol";
    const h3 = document.createElement("h3");
    h3.className = "studentLabel";
    h3.textContent = p.label;
    const pre = document.createElement("pre");
    pre.className = "textBlock pre";
    if (p.text == null) {
      pre.textContent = "（无预测）";
      pre.classList.add("empty");
    } else {
      pre.textContent = p.text;
    }
    col.appendChild(h3);
    col.appendChild(pre);
    container.appendChild(col);
  }
}

function render() {
  renderSummary();
  const position = document.getElementById("position");
  if (!state.items.length) {
    position.textContent = "当前问题：0 / 0";
    clearDetail();
    return;
  }

  const item = state.items[state.index];
  position.textContent = `当前问题：${state.index + 1} / ${state.items.length}`;
  document.getElementById("image").src = item.image_url;
  document.getElementById("question").textContent = item.question;
  document.getElementById("meta").textContent = JSON.stringify(item.meta || {}, null, 2);

  // 审批模式: 教师答案 + 审批状态
  document.getElementById("answerReview").textContent = item.answer;
  document.getElementById("reviewStatus").textContent = item.review
    ? `当前状态: ${item.review.decision === "approved" ? "通过" : "不通过"}`
    : "当前状态: 未审批";
  document.getElementById("note").value = item.review?.note || "";

  // 对比模式: 多模型并排
  renderCompareColumns(item);
}

async function saveReview(decision) {
  if (state.mode !== "review" || !state.items.length) return;
  const item = state.items[state.index];
  const note = document.getElementById("note").value;
  await fetch("/api/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: item.id, decision, note }),
  });
  await loadItems();
}

function move(offset) {
  if (!state.items.length) return;
  state.index = Math.min(state.items.length - 1, Math.max(0, state.index + offset));
  render();
}

async function setMode(mode) {
  state.mode = mode;
  state.index = 0;
  document.getElementById("app").className = `app mode-${mode}`;
  document.getElementById("modeReviewBtn").classList.toggle("active", mode === "review");
  document.getElementById("modeCompareBtn").classList.toggle("active", mode === "compare");
  await loadItems();
}

document.getElementById("modeReviewBtn").addEventListener("click", () => setMode("review"));
document.getElementById("modeCompareBtn").addEventListener("click", () => setMode("compare"));

document.getElementById("filter").addEventListener("change", async (e) => {
  state.filter = e.target.value;
  state.index = 0;
  await loadItems();
});

document.getElementById("prevBtn").addEventListener("click", () => move(-1));
document.getElementById("nextBtn").addEventListener("click", () => move(1));
document.getElementById("approveBtn").addEventListener("click", () => saveReview("approved"));
document.getElementById("rejectBtn").addEventListener("click", () => saveReview("rejected"));

loadItems();
