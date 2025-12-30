export function connectVoiceWS(
  onJson: (payload: any) => void,
  onBinary: (data: ArrayBuffer) => void
): WebSocket {
  const ws = new WebSocket("ws://localhost:8000/ws");
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => {
    ws.send(
      JSON.stringify({
        type: "START_SESSION",
        user: { id: "u1", name: "John" },
        language: "en",
      })
    );
  });

  ws.addEventListener("message", (event) => {
    if (typeof event.data === "string") {
      try {
        const payload = JSON.parse(event.data);
        onJson(payload);
      } catch (err) {
        console.warn("Failed to parse JSON message", err);
      }
      return;
    }

    if (event.data instanceof ArrayBuffer) {
      onBinary(event.data);
      return;
    }

    if (event.data instanceof Blob) {
      event.data.arrayBuffer().then(onBinary).catch((err) => {
        console.error("Failed to read binary message", err);
      });
    }
  });

  return ws;
}
