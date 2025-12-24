import { useEffect, useRef, useState } from 'react'
import './App.css'
import { playWavBytes } from './voice/audio'
import { startMicStreaming } from './voice/mic'
import { connectVoiceWS } from './voice/ws'

type LedgerItem = {
  id: string
  title: string
  to_label: string
  amount_lkr: number
  note?: string
  timestamp: string
}

type Screen = 'home' | 'transfer' | 'confirm' | 'success' | 'history'

function App() {
  const [assistantText, setAssistantText] = useState<string>('')
  const [connected, setConnected] = useState<boolean>(false)
  const [streaming, setStreaming] = useState<boolean>(false)
  const [audioStats, setAudioStats] = useState<{ total_bytes: number; seconds_estimate: number } | null>(null)
  const [transcripts, setTranscripts] = useState<string[]>([])
  const [lastAsrAt, setLastAsrAt] = useState<number>(0)
  const [screen, setScreen] = useState<Screen>('home')
  const [recipient, setRecipient] = useState<string>('')
  const [amount, setAmount] = useState<number | null>(null)
  const [note, setNote] = useState<string>('')
  const [confirmSummary, setConfirmSummary] = useState<string>('')
  const [biometricPrompt, setBiometricPrompt] = useState<boolean>(false)
  const [history, setHistory] = useState<LedgerItem[]>([])
  const [toast, setToast] = useState<string>('')

  const expectingAudio = useRef(false)
  const wsRef = useRef<WebSocket | null>(null)
  const stopMicRef = useRef<(() => void) | null>(null)

  const applyUiActions = (actions: any[]) => {
    actions.forEach((action) => {
      if (!action || typeof action !== 'object') return
      switch (action.type) {
        case 'NAVIGATE':
          setScreen(action.screen)
          if (action.screen === 'success') {
            setBiometricPrompt(false)
          }
          break
        case 'SET_FIELD':
          if (action.field === 'recipient') setRecipient(action.value ?? '')
          if (action.field === 'amount') setAmount(action.value ?? null)
          if (action.field === 'note') setNote(action.value ?? '')
          break
        case 'SHOW_CONFIRM':
          setConfirmSummary(action.summary ?? '')
          break
        case 'PROMPT_BIOMETRIC':
          setBiometricPrompt(true)
          break
        case 'SHOW_HISTORY':
          setHistory(Array.isArray(action.items) ? action.items : [])
          break
        case 'SHOW_TOAST':
          setToast(action.message ?? '')
          break
        default:
          break
      }
    })
  }

  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(() => setToast(''), 2200)
    return () => clearTimeout(timer)
  }, [toast])

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
    if (data.type === 'ASR_FINAL') {
      const text = data.text ?? ''
      if (text) {
        setTranscripts((prev) => [...prev, text])
        setLastAsrAt(Date.now())
      }
    }
    if (data.type === 'UI_ACTIONS' && Array.isArray(data.actions)) {
      applyUiActions(data.actions)
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
    setTranscripts([])
    setAudioStats(null)
    setAssistantText('')
    setScreen('home')
    setRecipient('')
    setAmount(null)
    setNote('')
    setConfirmSummary('')
    setHistory([])
    setBiometricPrompt(false)
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
      setBiometricPrompt(false)
    }
  }

  const handleStop = () => {
    stopMicRef.current?.()
    stopMicRef.current = null
    setStreaming(false)
    setBiometricPrompt(false)
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'STOP' }))
      wsRef.current.close()
    }
  }

  const sendBiometricResult = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    wsRef.current.send(JSON.stringify({ type: 'BIOMETRIC_RESULT', ok: true }))
    setBiometricPrompt(false)
  }

  return (
    <div className="app-shell">
      <header>
        <h1>TrustiPay Voice Prototype</h1>
        <p>Milestone D — Deterministic transfer flow</p>
      </header>

      <main>
        <div className="controls">
          <button className="start-button" onClick={handleStart} disabled={connected}>
            {connected ? 'Connected' : 'Start'}
          </button>
          <button className="stop-button" onClick={handleStop} disabled={!connected}>
            Stop
          </button>
          <span className={`listening-pill ${streaming ? 'on' : 'off'}`}>{streaming ? 'Listening…' : 'Idle'}</span>
          <span className="screen-chip">Screen: {screen}</span>
        </div>

        {toast && <div className="toast">{toast}</div>}

        <div className="grid">
          <div className="panel">
            <h2>Assistant</h2>
            <p className="assistant-text">{assistantText || 'Waiting for greeting...'}</p>
          </div>

          <div className="panel">
            <h2>Audio Stats</h2>
            <p>Streaming: {streaming ? 'Yes' : 'No'}</p>
            <p>
              Total bytes: {audioStats ? audioStats.total_bytes : 0} ({audioStats ? audioStats.seconds_estimate : 0} sec
              est.)
            </p>
          </div>

          <div className="panel transcript-panel">
            <div className="transcript-header">
              <h2>ASR Transcript</h2>
              <span className={`asr-indicator ${Date.now() - lastAsrAt < 1200 ? 'flash' : ''}`}>Final</span>
            </div>
            {transcripts.length === 0 ? (
              <p className="assistant-text">Speak to see transcripts here.</p>
            ) : (
              <ul>
                {transcripts.map((t, idx) => (
                  <li key={`${idx}-${t.slice(0, 8)}`}>{t}</li>
                ))}
              </ul>
            )}
          </div>

          <div className="panel flow-panel">
            <h2>Transfer</h2>
            <div className="field-row">
              <span>Recipient</span>
              <strong>{recipient || '—'}</strong>
            </div>
            <div className="field-row">
              <span>Amount</span>
              <strong>{amount ?? '—'}</strong>
            </div>
            <div className="field-row">
              <span>Note</span>
              <strong>{note || '—'}</strong>
            </div>
          </div>

          <div className="panel confirm-panel">
            <h2>Confirm</h2>
            <p>{confirmSummary || 'Waiting for summary...'}</p>
            {biometricPrompt && (
              <button className="fingerprint-btn" onClick={sendBiometricResult}>
                🔒 Approve with fingerprint
              </button>
            )}
          </div>

          <div className="panel success-panel">
            <h2>Success / History</h2>
            {history.length === 0 ? (
              <p>No history yet.</p>
            ) : (
              <ul className="history-list">
                {history.map((item) => (
                  <li key={item.id}>
                    <div className="history-title">{item.title}</div>
                    <div className="history-meta">
                      LKR {item.amount_lkr} • {item.to_label}
                    </div>
                    {item.note && <div className="history-note">Note: {item.note}</div>}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
