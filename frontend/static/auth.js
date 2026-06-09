// Shared helpers for the login / register pages.

function $(id) { return document.getElementById(id); }

function isValidEmail(e) { return /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(e); }

function fieldError(name, msg) {
  const el = document.querySelector(`[data-err="${name}"]`);
  if (el) { el.textContent = msg; el.classList.remove("hidden"); }
}

function clearErrors() {
  document.querySelectorAll("[data-err]").forEach((e) => {
    e.textContent = ""; e.classList.add("hidden");
  });
  const f = $("error");
  if (f) { f.classList.add("hidden"); f.textContent = ""; }
}

function formError(msg) {
  const f = $("error");
  if (f) { f.textContent = msg; f.classList.remove("hidden"); }
}

function setLoading(on, label) {
  const btn = $("submitBtn"), sp = $("spinner"), lbl = $("btnLabel");
  if (btn) btn.disabled = on;
  if (sp) sp.classList.toggle("hidden", !on);
  if (lbl && label) lbl.textContent = label;
}

async function postForm(url, obj) {
  const fd = new FormData();
  Object.entries(obj).forEach(([k, v]) => fd.append(k, v));
  try {
    return await fetch(url, { method: "POST", body: fd });
  } catch {
    throw new Error("Couldn't reach the server. Please try again.");
  }
}

const _EYE =
  '<svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" ' +
  'stroke="currentColor" stroke-width="1.6"><path stroke-linecap="round" stroke-linejoin="round" ' +
  'd="M2.25 12s3.75-7.5 9.75-7.5 9.75 7.5 9.75 7.5-3.75 7.5-9.75 7.5S2.25 12 2.25 12z"/>' +
  '<circle cx="12" cy="12" r="3"/></svg>';
const _EYE_OFF =
  '<svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" ' +
  'stroke="currentColor" stroke-width="1.6"><path stroke-linecap="round" stroke-linejoin="round" ' +
  'd="M3 3l18 18M10.6 10.6a3 3 0 004.24 4.24M9.9 4.24A9.5 9.5 0 0112 4c6 0 9.75 8 9.75 8a17 17 0 ' +
  '01-3.2 4.1M6.1 6.1A17 17 0 002.25 12S6 19.5 12 19.5c.9 0 1.76-.12 2.57-.34"/></svg>';

function initPasswordToggles() {
  document.querySelectorAll(".pw-toggle").forEach((btn) => {
    btn.innerHTML = _EYE;
    btn.addEventListener("click", () => {
      const inp = document.getElementById(btn.dataset.target);
      if (!inp) return;
      const show = inp.type === "password";
      inp.type = show ? "text" : "password";
      btn.innerHTML = show ? _EYE_OFF : _EYE;
      btn.setAttribute("aria-label", show ? "Hide password" : "Show password");
    });
  });
}
