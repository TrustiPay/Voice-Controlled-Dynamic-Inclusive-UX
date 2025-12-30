export async function playWavBytes(buffer: ArrayBuffer) {
  const blob = new Blob([buffer], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);

  const audio = new Audio(url);
  try {
    await audio.play();
  } finally {
    // Allow the browser to reclaim the URL once playback finishes
    audio.addEventListener("ended", () => URL.revokeObjectURL(url), {
      once: true,
    });
  }
}
