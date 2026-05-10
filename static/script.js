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
  const reportImage = document.getElementById("report-image");
  const reportLink = document.getElementById("report-link");
  const coachPanel = document.getElementById("coach-panel");
  const coachScore = document.getElementById("coach-score");
  const statImpactSpeed = document.getElementById("stat-impact-speed");
  const statBackswing = document.getElementById("stat-backswing");
  const statTorsoStability = document.getElementById("stat-torso-stability");
  const coachStrength = document.getElementById("coach-strength");
  const coachWeakness = document.getElementById("coach-weakness");
  const coachAdvice = document.getElementById("coach-advice");
  const drill1Video = document.getElementById("drill1-video");
  const drill1Badge = document.getElementById("drill1-badge");
  const drill1Title = document.getElementById("drill1-title");
  const drill1Why = document.getElementById("drill1-why");
  const drill1Text = document.getElementById("drill1-text");
  const drill2Video = document.getElementById("drill2-video");
  const drill2Badge = document.getElementById("drill2-badge");
  const drill2Title = document.getElementById("drill2-title");
  const drill2Why = document.getElementById("drill2-why");
  const drill2Text = document.getElementById("drill2-text");
  const schedulePrematch = document.getElementById("schedule-prematch");
  const scheduleRestday = document.getElementById("schedule-restday");
  const analysisTimeBadge = document.getElementById("analysis-time-badge");
  const playAudioBtn = document.getElementById("play-audio-btn");
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
  let latestCoachingAudioText = "";

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

  function fmtImpactSpeed(v) {
    const n = Number(v);
    return Number.isFinite(n) && n > 1e-3 ? `${n.toFixed(1)} px/s` : "Analysis pending";
  }

  function fmtBackswingDeg(v) {
    const n = Number(v);
    return Number.isFinite(n) && n > 1e-3 ? `${n.toFixed(1)}°` : "Analysis pending";
  }

  function fmtTorsoStability(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "Analysis pending";
    if (n <= 1e-6) return "Optimal range";
    return `${n.toFixed(2)}°`;
  }

  /** Normalize drill reference to https://www.youtube.com/embed/ID */
  function youtubeEmbedFromId(raw) {
    if (raw == null || raw === "") return "about:blank";
    const s = String(raw).trim();
    if (/^[\w-]{11}$/.test(s)) return `https://www.youtube.com/embed/${s}`;
    let ustr = s;
    if (ustr.startsWith("//")) ustr = `https:${ustr}`;
    else if (!/^https?:/i.test(ustr)) ustr = `https://${ustr}`;
    try {
      const u = new URL(ustr);
      const host = u.hostname.replace(/^www\./, "");
      if (host === "youtu.be") {
        const id = u.pathname.replace(/^\//, "").split("/")[0];
        if (/^[\w-]{11}$/.test(id)) return `https://www.youtube.com/embed/${id}`;
      }
      if (host.includes("youtube.com")) {
        const v = u.searchParams.get("v");
        if (v && /^[\w-]{11}$/.test(v)) return `https://www.youtube.com/embed/${v}`;
        const m = u.pathname.match(/\/embed\/([\w-]{11})/);
        if (m) return `https://www.youtube.com/embed/${m[1]}`;
      }
    } catch (_) {
      /* ignore */
    }
    return "about:blank";
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
    if (reportImage) {
      reportImage.removeAttribute("src");
      reportImage.hidden = true;
    }
    if (reportLink) {
      reportLink.removeAttribute("href");
      reportLink.hidden = true;
    }
    if (coachPanel) coachPanel.hidden = true;
    if (coachScore) coachScore.textContent = "--%";
    if (statImpactSpeed) statImpactSpeed.textContent = "--";
    if (statBackswing) statBackswing.textContent = "--";
    if (statTorsoStability) statTorsoStability.textContent = "--";
    if (coachStrength) coachStrength.textContent = "✅ --";
    if (coachWeakness) coachWeakness.textContent = "❌ --";
    if (coachAdvice) coachAdvice.textContent = "💡 --";
    if (drill1Video) drill1Video.src = "about:blank";
    if (drill1Badge) drill1Badge.textContent = "Primary focus";
    if (drill1Title) drill1Title.textContent = "--";
    if (drill1Why) drill1Why.textContent = "Rationale —";
    if (drill1Text) drill1Text.textContent = "--";
    if (drill2Video) drill2Video.src = "about:blank";
    if (drill2Badge) drill2Badge.textContent = "Supporting focus";
    if (drill2Title) drill2Title.textContent = "--";
    if (drill2Why) drill2Why.textContent = "Rationale —";
    if (drill2Text) drill2Text.textContent = "--";
    if (schedulePrematch) schedulePrematch.textContent = "--";
    if (scheduleRestday) scheduleRestday.textContent = "--";
    if (analysisTimeBadge) analysisTimeBadge.textContent = "Processed in --s";
    latestCoachingAudioText = "";
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
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
      const videoName = data.video_filename || data.filename;
      if (!videoName) {
        showError("Missing result filename in response.");
        return;
      }
      const reportName = data.report_filename;
      const videoUrl = data.video_url || `/results/${encodeURIComponent(videoName)}`;
      const reportUrl = data.report_url || (reportName ? `/results/${encodeURIComponent(reportName)}` : "");
      const coaching = data.coaching_data || {};
      resultVideo.src = videoUrl;
      if (reportImage && reportName) {
        reportImage.src = reportUrl;
        reportImage.hidden = false;
      }
      if (reportLink && reportName) {
        reportLink.href = reportUrl;
        reportLink.hidden = false;
      }
      if (coachPanel) {
        coachPanel.hidden = false;
        const score = Number(coaching.overall_form_score);
        if (coachScore) coachScore.textContent = Number.isFinite(score) ? `${Math.round(score)}%` : "--%";
        const stats = coaching.key_stats || {};
        if (statImpactSpeed) statImpactSpeed.textContent = fmtImpactSpeed(stats.impact_speed);
        if (statBackswing) statBackswing.textContent = fmtBackswingDeg(stats.max_backswing_angle);
        if (statTorsoStability) statTorsoStability.textContent = fmtTorsoStability(stats.torso_stability);
        const strengths = Array.isArray(coaching.strengths) ? coaching.strengths : [];
        const weaknesses = Array.isArray(coaching.weaknesses) ? coaching.weaknesses : [];
        if (coachStrength) coachStrength.textContent = `✅ ${strengths[0] || "--"}`;
        if (coachWeakness) coachWeakness.textContent = `❌ ${weaknesses[0] || "--"}`;
        if (coachAdvice) coachAdvice.textContent = `💡 ${coaching.actionable_advice || "--"}`;
        const drills = Array.isArray(coaching.recommended_drills) ? coaching.recommended_drills : [];
        const d1 = drills[0] || {};
        const d2 = drills[1] || {};
        if (drill1Video) drill1Video.src = youtubeEmbedFromId(d1.video_id);
        if (drill1Badge) drill1Badge.textContent = d1.priority_badge || "Primary focus";
        if (drill1Title) drill1Title.textContent = d1.title || "--";
        if (drill1Why) drill1Why.textContent = `Rationale: ${d1.why_text || "—"}`;
        if (drill1Text) drill1Text.textContent = d1.instruction || "--";
        if (drill2Video) drill2Video.src = youtubeEmbedFromId(d2.video_id);
        if (drill2Badge) drill2Badge.textContent = d2.priority_badge || "Supporting focus";
        if (drill2Title) drill2Title.textContent = d2.title || "--";
        if (drill2Why) drill2Why.textContent = `Rationale: ${d2.why_text || "—"}`;
        if (drill2Text) drill2Text.textContent = d2.instruction || "--";
        const pres = d1.prescription || {};
        if (schedulePrematch) schedulePrematch.textContent = pres.pre_match || "--";
        if (scheduleRestday) scheduleRestday.textContent = pres.rest_day || "--";
        if (analysisTimeBadge) {
          const t = Number(coaching.analysis_time_sec);
          analysisTimeBadge.textContent = Number.isFinite(t) ? `Processed in ${t.toFixed(2)}s` : "Processed in --s";
        }
        latestCoachingAudioText = typeof coaching.coaching_audio_text === "string" ? coaching.coaching_audio_text : "";
      }
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

  if (playAudioBtn) {
    playAudioBtn.addEventListener("click", () => {
      if (!latestCoachingAudioText) return;
      if (!("speechSynthesis" in window)) {
        showError("Speech synthesis is not available in this browser.");
        return;
      }
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(latestCoachingAudioText);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    });
  }

  fileInput.addEventListener("change", () => {
    if (fileLabel) {
      fileLabel.textContent = fileInput.files && fileInput.files[0]
        ? fileInput.files[0].name
        : "Select video…";
    }
  });

  setProcessing(false);
})();
