const SAMPLE_RATE = 16000
const BUFFER_SIZE = 2048

function floatTo16BitPCM(input: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(input.length * 2)
  const view = new DataView(buffer)
  for (let i = 0; i < input.length; i++) {
    // Clamp the sample
    let s = Math.max(-1, Math.min(1, input[i]))
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return buffer
}

export async function startMicStreaming(ws: WebSocket): Promise<{ stop: () => void }> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

  const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE })
  const source = audioContext.createMediaStreamSource(stream)

  const processor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1)
  processor.onaudioprocess = (event: AudioProcessingEvent) => {
    if (ws.readyState !== WebSocket.OPEN) return
    const inputBuffer = event.inputBuffer.getChannelData(0)
    const pcmBuffer = floatTo16BitPCM(inputBuffer)
    ws.send(pcmBuffer)
  }

  source.connect(processor)
  processor.connect(audioContext.destination)

  const stop = () => {
    processor.disconnect()
    source.disconnect()
    processor.onaudioprocess = null
    stream.getTracks().forEach((t) => t.stop())
    audioContext.close().catch(() => undefined)
  }

  return { stop }
}
