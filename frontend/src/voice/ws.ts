export type JsonHandler = (data: any) => void
export type BinaryHandler = (data: ArrayBuffer) => void
export type OpenHandler = () => void

export function connectVoiceWS(onJson: JsonHandler, onBinary: BinaryHandler, onOpen?: OpenHandler): WebSocket {
  const ws = new WebSocket('ws://localhost:8000/ws')
  ws.binaryType = 'arraybuffer'

  ws.onopen = () => {
    const startPayload = {
      type: 'START_SESSION',
      user: { id: 'u1', name: 'John' },
      language: 'en',
    }
    ws.send(JSON.stringify(startPayload))
    onOpen?.()
  }

  ws.onmessage = (event: MessageEvent) => {
    const { data } = event
    if (typeof data === 'string') {
      try {
        const parsed = JSON.parse(data)
        onJson(parsed)
      } catch (err) {
        console.error('Failed to parse JSON from WS', err)
      }
      return
    }

    // Blob or ArrayBuffer
    if (data instanceof ArrayBuffer) {
      onBinary(data)
      return
    }

    if (data instanceof Blob) {
      data.arrayBuffer().then(onBinary).catch((err) => console.error('Blob to ArrayBuffer failed', err))
    }
  }

  return ws
}
