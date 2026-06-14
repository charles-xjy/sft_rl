const state = {
  items: [],
  index: 0,
  filter: "pending",
};

async function loadItems() {
  const res = await fetch(`/api/items?filter=${encodeURIComponent(state.filter)}`);
  const data = await res.json();
  state.items = data.items;
  state.summary = data.summary;
  if (state.index >= state.items.length) state.index = Math.max(0, state.items.length - 1);
  render();
}

function renderSummary() {
  const s = state.summary || { total: 0, reviewed: 0, approved: 0, rejected: 0 };
  document.getElementById("summary").innerHTML = `
    <p>总数: <strong>${s.total}</strong></p>
    <p>已审: <strong>${s.reviewed}</strong></p>
    <p>通过: <strong>${s.approved}</strong></p>
    <p>不通过: <strong>${s.rejected}</strong></p>
  `;
}

function render() {
  renderSummary();
  const position = document.getElementById("position");
  if (!state.items.length) {
    position.textContent = "当前问题：0 / 0";
    document.getElementById("image").src = "";
    document.getElementById("question").textContent = "";
    document.getElementById("answer").textContent = "";
    document.getElementById("meta").textContent = "";
    document.getElementById("reviewStatus").textContent = "";
    document.getElementById("note").value = "";
    return;
  }

  const item = state.items[state.index];
  position.textContent = `当前问题：${state.index + 1} / ${state.items.length}`;
  document.getElementById("image").src = item.image_url;
  document.getElementById("question").textContent = item.question;
  document.getElementById("answer").textContent = item.answer;
  document.getElementById("meta").textContent = JSON.stringify(item.meta?.auto_validation || {}, null, 2);
  document.getElementById("note").value = item.review?.note || "";
  document.getElementById("reviewStatus").textContent = item.review
    ? `当前状态: ${item.review.decision === "approved" ? "通过" : "不通过"}`
    : "当前状态: 未审批";
}

async function saveReview(decision) {
  if (!state.items.length) return;
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

document.getElementById("filter").addEventListener("change", async (e) => {
  state.filter = e.target.value;
  state.index = 0;
  await loadItems();
});

document.getElementById("prevBtn").addEventListener("click", () => move(-1));
document.getElementById("nextBtn").addEventListener("click", () => move(1));
document.getElementById("approveBtn").addEventListener("click", async () => {
  await saveReview("approved");
});
document.getElementById("rejectBtn").addEventListener("click", async () => {
  await saveReview("rejected");
});

loadItems();
