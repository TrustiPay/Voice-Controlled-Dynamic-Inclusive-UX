const SAMPLE_RATE = 16000;
const CHANNELS = 1;
const BYTES_PER_SAMPLE = 2; // int16

function floatTo16BitPCM(float32: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(float32.length * BYTES_PER_SAMPLE);
  const view = new DataView(buffer);

  for (let i = 0; i < float32.length; i += 1) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(i * BYTES_PER_SAMPLE, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return buffer;
}

export async function startMicStreaming(
  ws: WebSocket
): Promise<{ stop: () => void }> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: CHANNELS,
      sampleRate: SAMPLE_RATE,
    },
    video: false,
  });

  const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioContext.resume();

  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, CHANNELS, CHANNELS);

  // Avoid audible loopback by routing through a silent gain node.
  const silence = audioContext.createGain();
  silence.gain.value = 0;

  processor.onaudioprocess = (event) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    const input = event.inputBuffer.getChannelData(0);
    const pcmBuffer = floatTo16BitPCM(input);
    ws.send(pcmBuffer);
  };

  source.connect(processor);
  processor.connect(silence);
  silence.connect(audioContext.destination);

  const stop = () => {
    processor.onaudioprocess = null;
    try {
      processor.disconnect();
      source.disconnect();
      silence.disconnect();
    } catch {
      // ignore
    }
    stream.getTracks().forEach((track) => track.stop());
    audioContext.close().catch(() => {});
  };

  return { stop };
}
