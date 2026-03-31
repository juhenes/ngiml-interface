document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("[data-predict-form]");
  const resultContainer = document.querySelector("#result-container");
  const submitButton = document.querySelector("[data-submit-button]");
  const title = document.querySelector("[data-result-title]");
  const errorBox = document.querySelector("[data-result-error]");

  if (!form || !resultContainer || !submitButton || !title || !errorBox) {
    return;
  }

  const previewKeys = ["input", "prediction", "binary", "overlay"];

  function setLoadingState(isLoading) {
    resultContainer.classList.toggle("is-loading", isLoading);
    submitButton.disabled = isLoading;
    submitButton.textContent = isLoading ? "Running..." : "Run Inference";
  }

  function resetError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function resetPreviews(message) {
    for (const key of previewKeys) {
      const placeholder = document.querySelector(`[data-preview-placeholder="${key}"]`);
      const image = document.querySelector(`[data-preview-image="${key}"]`);
      if (placeholder) {
        placeholder.textContent = message;
        placeholder.classList.remove("hidden");
      }
      if (image) {
        image.removeAttribute("src");
        image.classList.add("hidden");
      }
    }
  }

  function applyPreview(key, url) {
    const placeholder = document.querySelector(`[data-preview-placeholder="${key}"]`);
    const image = document.querySelector(`[data-preview-image="${key}"]`);
    if (!placeholder || !image) {
      return;
    }
    image.onload = () => {
      placeholder.classList.add("hidden");
      image.classList.remove("hidden");
    };
    image.src = url;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    resetError();
    title.textContent = "Running inference...";
    resetPreviews("Preparing preview...");
    setLoadingState(true);

    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),
      });
      const payload = await response.json();

      if (!response.ok || !payload.ok) {
        title.textContent = "Prediction failed";
        showError(payload.error || "Request failed. Please try again.");
        resetPreviews("No preview available");
        return;
      }

      title.textContent = payload.checkpointLabel;
      for (const key of previewKeys) {
        applyPreview(key, payload.previews[key]);
      }
    } catch (error) {
      title.textContent = "Prediction failed";
      showError("Request failed. Please try again.");
      resetPreviews("No preview available");
    } finally {
      setLoadingState(false);
    }
  });
});
