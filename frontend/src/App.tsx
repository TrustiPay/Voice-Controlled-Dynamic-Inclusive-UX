import { useRef, useState } from 'react'
import './App.css'
import { playWavBytes } from './voice/audio'
import { startMicStreaming } from './voice/mic'
import { connectVoiceWS } from './voice/ws'

function App() {
  const [assistantText, setAssistantText] = useState<string>('')
  const [connected, setConnected] = useState<boolean>(false)
  const [streaming, setStreaming] = useState<boolean>(false)
  const [audioStats, setAudioStats] = useState<{ total_bytes: number; seconds_estimate: number } | null>(null)
  const expectingAudio = useRef(false)
  const wsRef = useRef<WebSocket | null>(null)
  const stopMicRef = useRef<(() => void) | null>(null)

  const handleJson = (data: any) => {
    if (!data || typeof data !== 'object') return
    if (data.type === 'AGENT_MESSAGE') {
      setAssistantText(data.text ?? '')
    }
    if (data.type === 'TTS_AUDIO') {
      expectingAudio.current = true
    }
    if (data.type === 'AUDIO_STATS') {
      setAudioStats({ total_bytes: data.total_bytes, seconds_estimate: data.seconds_estimate })
    }
  }

  const handleBinary = (bytes: ArrayBuffer) => {
    if (expectingAudio.current) {
      expectingAudio.current = false
      playWavBytes(bytes).catch((err) => console.error('Audio playback failed', err))
    }
  }

  const handleStart = async () => {
    if (wsRef.current) return
    wsRef.current = connectVoiceWS(handleJson, handleBinary, async () => {
      setConnected(true)
      wsRef.current?.send(
        JSON.stringify({
          type: 'AUDIO_CONFIG',
          format: 'pcm_s16le',
          sample_rate: 16000,
          channels: 1,
        })
      )
      try {
        const { stop } = await startMicStreaming(wsRef.current!)
        stopMicRef.current = stop
        setStreaming(true)
      } catch (err) {
        console.error('Mic streaming failed', err)
      }
    })
    wsRef.current.onclose = () => {
      wsRef.current = null
      setConnected(false)
      setStreaming(false)
      stopMicRef.current?.()
      stopMicRef.current = null
    }
  }

  const handleStop = () => {
    stopMicRef.current?.()
    stopMicRef.current = null
    setStreaming(false)
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'STOP' }))
      wsRef.current.close()
    }
  }

  return (
    <div className="app-shell">
      <header>
        <h1>TrustiPay Voice Prototype</h1>
        <p>Milestone B — Mic streaming + audio stats</p>
      </header>

      <main>
        <div className="controls">
          <button className="start-button" onClick={handleStart} disabled={connected}>
            {connected ? 'Connected' : 'Start'}
          </button>
          <button className="stop-button" onClick={handleStop} disabled={!connected}>
            Stop
          </button>
        </div>

        <div className="message-panel">
          <h2>Assistant</h2>
          <p className="assistant-text">{assistantText || 'Waiting for greeting...'}</p>
        </div>

        <div className="stats-panel">
          <h2>Audio Stats</h2>
          <p>Streaming: {streaming ? 'Yes' : 'No'}</p>
          <p>
            Total bytes: {audioStats ? audioStats.total_bytes : 0} ({audioStats ? audioStats.seconds_estimate : 0} sec est.)
          </p>
        </div>
      </main>
    </div>
  )
}

export default App
