/**
 * Strike Lab — upload, analyze, play result (expects Flask app.py on same origin).
 */
(function () {
  const fileInput = document.getElementById("strike-file");
  const analyzeBtn = document.getElementById("analyze-btn");
  const uploadCard = document.getElementById("upload-card");
  const overlay = document.getElementById("processing-overlay");
  const statusLine1 = document.getElementById("status-line-1");
  const statusLine2 = document.getElementById("status-line-2");
  const resultSection = document.getElementById("result-section");
  const resultVideo = document.getElementById("result-video");
  const resetBtn = document.getElementById("reset-btn");
  const errorBanner = document.getElementById("error-banner");
  const fileLabel = document.querySelector('label[for="strike-file"]');

  const STATUS_MESSAGES = [
    [
      "Utilizing AMD Instinct MI300X for Biomechanical Inference…",
      "Computing vector displacement fields…",
    ],
    [
      "MediaPipe Pose · model complexity 2",
      "Savitzky–Golay temporal smoothing…",
    ],
    [
      "Aligning ghost strike manifold to peak ankle velocity…",
      "Rendering neon biomechanics overlay…",
    ],
  ];

  let statusInterval = null;

  function showError(msg) {
    errorBanner.textContent = msg || "Something went wrong.";
    errorBanner.hidden = false;
  }

  function clearError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
  }

  function setProcessing(active) {
    if (overlay) {
      overlay.classList.toggle("processing-overlay--visible", active);
      overlay.setAttribute("aria-hidden", active ? "false" : "true");
    }
    analyzeBtn.disabled = active;
    fileInput.disabled = active;
    if (active) {
      let i = 0;
      statusInterval = window.setInterval(() => {
        const pair = STATUS_MESSAGES[i % STATUS_MESSAGES.length];
        statusLine1.textContent = pair[0];
        statusLine2.textContent = pair[1];
        i += 1;
      }, 2200);
      statusLine1.textContent = STATUS_MESSAGES[0][0];
      statusLine2.textContent = STATUS_MESSAGES[0][1];
    } else if (statusInterval) {
      window.clearInterval(statusInterval);
      statusInterval = null;
    }
  }

  function resetUi() {
    setProcessing(false);
    clearError();
    fileInput.value = "";
    if (fileLabel) fileLabel.textContent = "Select video…";
    resultSection.hidden = true;
    uploadCard.hidden = false;
    analyzeBtn.hidden = false;
    resultVideo.removeAttribute("src");
    resultVideo.load();
  }

  analyzeBtn.addEventListener("click", async () => {
    clearError();
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      showError("Choose an .mp4 or .mov file first.");
      return;
    }

    const fd = new FormData();
    fd.append("video", file);

    setProcessing(true);
    try {
      const res = await fetch("/analyze-video", {
        method: "POST",
        body: fd,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        showError(data.error || `Server error (${res.status})`);
        return;
      }
      const name = data.filename;
      if (!name) {
        showError("Missing result filename in response.");
        return;
      }
      const url = `/results/${encodeURIComponent(name)}`;
      resultVideo.src = url;
      uploadCard.hidden = true;
      analyzeBtn.hidden = true;
      resultSection.hidden = false;
      try {
        await resultVideo.play();
      } catch (_) {
        /* autoplay may be blocked until user gesture — controls still work */
      }
    } catch (e) {
      showError(e instanceof Error ? e.message : String(e));
    } finally {
      setProcessing(false);
    }
  });

  resetBtn.addEventListener("click", resetUi);

  fileInput.addEventListener("change", () => {
    if (fileLabel) {
      fileLabel.textContent = fileInput.files && fileInput.files[0]
        ? fileInput.files[0].name
        : "Select video…";
    }
  });

  setProcessing(false);
})();
