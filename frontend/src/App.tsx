import { useRef, useState } from 'react'
import './App.css'
import { playWavBytes } from './voice/audio'
import { connectVoiceWS } from './voice/ws'

function App() {
  const [assistantText, setAssistantText] = useState<string>('')
  const [connected, setConnected] = useState<boolean>(false)
  const expectingAudio = useRef(false)
  const wsRef = useRef<WebSocket | null>(null)

  const handleJson = (data: any) => {
    if (!data || typeof data !== 'object') return
    if (data.type === 'AGENT_MESSAGE') {
      setAssistantText(data.text ?? '')
    }
    if (data.type === 'TTS_AUDIO') {
      expectingAudio.current = true
    }
  }

  const handleBinary = (bytes: ArrayBuffer) => {
    if (expectingAudio.current) {
      expectingAudio.current = false
      playWavBytes(bytes).catch((err) => console.error('Audio playback failed', err))
    }
  }

  const handleStart = () => {
    if (wsRef.current) return
    wsRef.current = connectVoiceWS(handleJson, handleBinary, () => setConnected(true))
    wsRef.current.onclose = () => {
      wsRef.current = null
      setConnected(false)
    }
  }

  return (
    <div className="app-shell">
      <header>
        <h1>TrustiPay Voice Prototype</h1>
        <p>Milestone A — WebSocket greeting + TTS beep</p>
      </header>

      <main>
        <button className="start-button" onClick={handleStart} disabled={connected}>
          {connected ? 'Connected' : 'Start'}
        </button>

        <div className="message-panel">
          <h2>Assistant</h2>
          <p className="assistant-text">{assistantText || 'Waiting for greeting...'}</p>
        </div>
      </main>
    </div>
  )
}

export default App
