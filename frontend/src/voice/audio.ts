export async function playWavBytes(buffer: ArrayBuffer): Promise<void> {
  const blob = new Blob([buffer], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);

  return new Promise((resolve, reject) => {
    const cleanup = () => URL.revokeObjectURL(url);
    audio.addEventListener(
      "ended",
      () => {
        cleanup();
        resolve();
      },
      { once: true }
    );
    audio.addEventListener(
      "error",
      () => {
        cleanup();
        reject(new Error("Audio playback error"));
      },
      { once: true }
    );
    audio.play().catch((err) => {
      cleanup();
      reject(err);
    });
  });
}
