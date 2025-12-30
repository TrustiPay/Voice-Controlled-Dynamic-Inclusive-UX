import { useEffect, useMemo, useRef, useState } from "react";
import { connectVoiceWS } from "./voice/ws";
import { playWavBytes } from "./voice/audio";
import { startMicStreaming } from "./voice/mic";
import "./index.css";

type AssistantMessage = {
  id: number;
  text: string;
};

type Screen = "home" | "transfer" | "confirm" | "success" | "history";

type HistoryItem = {
  id: string;
  title?: string;
  amount_lkr?: number;
  to_label?: string;
  timestamp?: string;
  note?: string | null;
};

function StatusBadge({ label }: { label: string }) {
  return (
    <span className="rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-semibold text-emerald-200">
      {label}
    </span>
  );
}

export default function App() {
  const [status, setStatus] = useState("Disconnected");
  const [assistantMessages, setAssistantMessages] = useState<AssistantMessage[]>(
    []
  );
  const [audioStatus, setAudioStatus] = useState("Idle");
  const [hasStarted, setHasStarted] = useState(false);
  const [micStatus, setMicStatus] = useState("Idle");
  const [audioBytes, setAudioBytes] = useState(0);
  const [audioSeconds, setAudioSeconds] = useState(0);
  const [transcripts, setTranscripts] = useState<string[]>([]);
  const [asrFlash, setAsrFlash] = useState(false);
  const [screen, setScreen] = useState<Screen>("home");
  const [recipientField, setRecipientField] = useState("");
  const [amountField, setAmountField] = useState("");
  const [noteField, setNoteField] = useState("");
  const [confirmSummary, setConfirmSummary] = useState("");
  const [biometricPrompt, setBiometricPrompt] = useState(false);
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [toast, setToast] = useState<string | null>(null);
  const expectingAudio = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);
  const stopMicRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    return () => {
      wsRef.current?.close(1000, "component unmounted");
    };
  }, []);

  const applyUiActions = (actions: any[]) => {
    if (!actions || !Array.isArray(actions)) return;
    actions.forEach((action) => {
      switch (action.type) {
        case "NAVIGATE":
          if (action.screen) {
            setScreen(action.screen);
          }
          if (action.screen === "success") {
            setBiometricPrompt(false);
          }
          break;
        case "SET_FIELD":
          if (action.field === "recipient") setRecipientField(action.value ?? "");
          if (action.field === "amount")
            setAmountField(
              action.value !== undefined && action.value !== null
                ? String(action.value)
                : ""
            );
          if (action.field === "note") setNoteField(action.value ?? "");
          break;
        case "SHOW_CONFIRM":
          if (action.summary) {
            setConfirmSummary(action.summary);
            setScreen("confirm");
          }
          break;
        case "PROMPT_BIOMETRIC":
          setBiometricPrompt(true);
          break;
        case "SHOW_HISTORY":
          if (Array.isArray(action.items)) {
            setHistoryItems(action.items);
          }
          break;
        case "SHOW_TOAST":
          if (action.message) {
            setToast(action.message);
            setTimeout(() => setToast(null), 2500);
          }
          break;
        default:
          break;
      }
    });
  };

  const sendBiometricOk = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: "BIOMETRIC_RESULT", ok: true }));
    setBiometricPrompt(false);
  };

  const greetingPlaceholder = useMemo(
    () => [
      "Press Start to begin a voice session.",
      "You should hear a short greeting clip.",
      "Messages from the assistant will show up here.",
    ],
    []
  );

  const renderScreen = () => {
    switch (screen) {
      case "home":
        return (
          <div className="space-y-2 text-slate-200">
            <p className="text-lg font-semibold text-white">Home</p>
            <p className="text-sm text-slate-300">
              Press Start and speak to begin a transfer.
            </p>
          </div>
        );
      case "transfer":
        return (
          <div className="space-y-4">
            <p className="text-lg font-semibold text-white">Transfer</p>
            <div className="space-y-2 text-sm">
              <label className="block text-slate-400">Recipient</label>
              <div className="rounded-xl bg-slate-900 px-3 py-2 ring-1 ring-slate-800">
                {recipientField || <span className="text-slate-500">Waiting…</span>}
              </div>
            </div>
            <div className="space-y-2 text-sm">
              <label className="block text-slate-400">Amount (LKR)</label>
              <div className="rounded-xl bg-slate-900 px-3 py-2 ring-1 ring-slate-800">
                {amountField || <span className="text-slate-500">Waiting…</span>}
              </div>
            </div>
            <div className="space-y-2 text-sm">
              <label className="block text-slate-400">Note</label>
              <div className="rounded-xl bg-slate-900 px-3 py-2 ring-1 ring-slate-800">
                {noteField || <span className="text-slate-500">Optional</span>}
              </div>
            </div>
          </div>
        );
      case "confirm":
        return (
          <div className="space-y-4">
            <p className="text-lg font-semibold text-white">Confirm</p>
            <div className="rounded-xl bg-slate-900 px-3 py-3 text-sm text-slate-100 ring-1 ring-slate-800">
              {confirmSummary || "Awaiting confirmation details…"}
            </div>
            {biometricPrompt && (
              <button
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-500 px-4 py-3 text-base font-semibold text-slate-900 transition hover:bg-emerald-400"
                onClick={sendBiometricOk}
              >
                <span role="img" aria-label="fingerprint">
                  🖐️
                </span>
                Approve with fingerprint
              </button>
            )}
          </div>
        );
      case "success":
        return (
          <div className="space-y-2">
            <p className="text-lg font-semibold text-white">Success</p>
            <p className="text-sm text-emerald-200">Payment sent</p>
          </div>
        );
      case "history":
        return (
          <div className="space-y-3">
            <p className="text-lg font-semibold text-white">History</p>
            {historyItems.length === 0 ? (
              <p className="text-sm text-slate-400">No history yet.</p>
            ) : (
              <div className="space-y-2">
                {historyItems.map((item) => (
                  <div
                    key={item.id}
                    className="rounded-xl bg-slate-900 px-3 py-2 text-sm text-slate-100 ring-1 ring-slate-800"
                  >
                    <div className="flex justify-between">
                      <span>{item.title || item.to_label}</span>
                      {item.amount_lkr !== undefined && (
                        <span className="font-semibold text-emerald-200">
                          LKR {item.amount_lkr}
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-slate-400">
                      {item.timestamp?.replace("T", " ")}
                    </div>
                    {item.note && (
                      <div className="text-xs text-slate-300">Note: {item.note}</div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      default:
        return null;
    }
  };

  const sendStopAndCleanup = ({ skipStop }: { skipStop?: boolean } = {}) => {
    stopMicRef.current?.();
    stopMicRef.current = null;
    expectingAudio.current = false;
    setMicStatus("Stopped");
    setHasStarted(false);
    setScreen("home");
    setBiometricPrompt(false);
    if (!skipStop) {
      try {
        wsRef.current?.send(JSON.stringify({ type: "STOP" }));
      } catch {
        // ignore send errors during teardown
      }
      wsRef.current?.close(1000, "user stop");
    }
    wsRef.current = null;
  };

  const handleStart = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }
    setStatus("Connecting…");
    setHasStarted(true);
    setMicStatus("Requesting mic…");
    setAudioBytes(0);
    setAudioSeconds(0);
    setTranscripts([]);
    setRecipientField("");
    setAmountField("");
    setNoteField("");
    setConfirmSummary("");
    setBiometricPrompt(false);
    setScreen("home");

    const ws = connectVoiceWS(
      (payload) => {
        if (payload?.type === "AGENT_MESSAGE") {
          setAssistantMessages((prev) => [
            ...prev,
            { id: Date.now() + prev.length, text: payload.text ?? "" },
          ]);
        } else if (payload?.type === "TTS_AUDIO") {
          expectingAudio.current = true;
        } else if (payload?.type === "AUDIO_STATS") {
          setAudioBytes(payload.total_bytes ?? 0);
          setAudioSeconds(payload.seconds_estimate ?? 0);
        } else if (payload?.type === "ASR_FINAL") {
          if (payload.text) {
            setTranscripts((prev) => [...prev, payload.text as string]);
            setAsrFlash(true);
            setTimeout(() => setAsrFlash(false), 600);
          }
        } else if (payload?.type === "UI_ACTIONS") {
          applyUiActions(payload.actions ?? []);
        }
      },
      async (data) => {
        if (!expectingAudio.current) return;
        expectingAudio.current = false;
        setAudioStatus("Playing audio…");
        try {
          await playWavBytes(data);
        } finally {
          setAudioStatus("Idle");
        }
      }
    );

    ws.addEventListener("open", async () => {
      setStatus("Connected");
      ws.send(
        JSON.stringify({
          type: "AUDIO_CONFIG",
          format: "pcm_s16le",
          sample_rate: 16000,
          channels: 1,
        })
      );

      try {
        const { stop } = await startMicStreaming(ws);
        stopMicRef.current = stop;
        setMicStatus("Streaming");
      } catch (err) {
        console.error("Mic error", err);
        setMicStatus("Mic error");
        sendStopAndCleanup();
      }
    });
    ws.addEventListener("close", () => {
      setStatus("Disconnected");
      sendStopAndCleanup({ skipStop: true });
    });
    ws.addEventListener("error", () => setStatus("Error"));
    wsRef.current = ws;
  };

  const handleStop = () => {
    sendStopAndCleanup();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <div className="mx-auto flex max-w-5xl flex-col gap-8 px-4 py-10">
        <header className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-3xl font-semibold leading-tight text-white">
              TrustiPay Voice Prototype
            </h1>
            <p className="text-sm text-slate-300">
              Milestone B — WS greeting + mic capture streaming (no ASR yet)
            </p>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge label={status} />
            <span className="rounded-full bg-slate-800 px-3 py-1 text-xs font-medium text-slate-200">
              Audio: {audioStatus}
            </span>
            <span className="rounded-full bg-slate-800 px-3 py-1 text-xs font-medium text-slate-200">
              Mic: {micStatus}
            </span>
          </div>
        </header>

        <div className="grid gap-6 md:grid-cols-[380px,1fr]">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-5 shadow-xl shadow-black/30 backdrop-blur">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.1em] text-slate-400">
                  Session
                </p>
                <p className="text-lg font-semibold text-white">Push to start</p>
              </div>
              <div className="flex items-center gap-2">
                {micStatus === "Streaming" && (
                  <span className="rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-semibold text-emerald-200 ring-1 ring-emerald-500/40">
                    Listening…
                  </span>
                )}
                <div className="h-10 w-10 rounded-full bg-emerald-500/20 ring-1 ring-emerald-400/60"></div>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <button
                className="rounded-xl bg-emerald-500 px-4 py-3 text-center text-base font-semibold text-slate-900 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
                disabled={hasStarted && status === "Connected"}
                onClick={handleStart}
              >
                {status === "Connected" ? "Session running" : "Start Voice Session"}
              </button>
              <button
                className="rounded-xl bg-slate-800 px-4 py-3 text-center text-base font-semibold text-slate-100 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-800/70 disabled:text-slate-500"
                onClick={handleStop}
                disabled={status === "Disconnected"}
              >
                Stop
              </button>
            </div>
            <p className="mt-2 text-xs text-slate-400">
              Sends START_SESSION + AUDIO_CONFIG, starts mic streaming at 16 kHz
              PCM, and waits for greeting audio.
            </p>

            <div className="mt-6 h-72 space-y-3 overflow-y-auto rounded-2xl bg-slate-950/60 p-4 ring-1 ring-slate-800">
              {assistantMessages.length === 0 ? (
                <div className="space-y-2 text-sm text-slate-400">
                  {greetingPlaceholder.map((line, idx) => (
                    <p key={idx}>{line}</p>
                  ))}
                </div>
              ) : (
                assistantMessages.map((msg) => (
                  <div
                    key={msg.id}
                    className="max-w-[90%] rounded-2xl bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100 ring-1 ring-emerald-500/30"
                  >
                    {msg.text}
                  </div>
                ))
              )}
            </div>
            <div className="mt-3 rounded-xl bg-slate-950/50 p-3 text-xs text-slate-300 ring-1 ring-slate-800">
              <div className="flex justify-between">
                <span>Audio bytes sent</span>
                <span>{audioBytes.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Approx. seconds</span>
                <span>{audioSeconds.toFixed(2)}</span>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-5 shadow-inner shadow-black/30 space-y-4">
            <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-white">App screen</h2>
                <span className="rounded-full bg-slate-800 px-3 py-1 text-xs font-semibold text-slate-200">
                  {screen}
                </span>
              </div>
              <div className="mt-3 rounded-xl bg-slate-900/80 p-4 ring-1 ring-slate-800">
                {renderScreen()}
              </div>
            </div>

            <div>
              <h2 className="text-lg font-semibold text-white">What this demo does</h2>
              <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-slate-200">
                <li>Connects to FastAPI /ws on localhost:8000.</li>
                <li>Sends a START_SESSION payload with the hardcoded user John.</li>
                <li>Declares AUDIO_CONFIG and streams 16 kHz pcm_s16le mic audio.</li>
                <li>Displays AGENT_MESSAGE bubbles from the backend.</li>
                <li>Plays WAV bytes streamed after the TTS_AUDIO header.</li>
                <li>Shows AUDIO_STATS from the backend as audio flows.</li>
                <li>Displays ASR_FINAL transcripts when VAD segments speech.</li>
                <li>Applies UI_ACTIONS to navigate, fill fields, and prompt biometric.</li>
              </ul>
              <div className="mt-4 rounded-2xl bg-slate-950/60 p-4 text-xs text-slate-400 ring-1 ring-slate-800">
                Tip: Run the backend with <code>uvicorn app:app --reload --port 8000</code>{" "}
                from <code>backend/</code>, then start the frontend via{" "}
                <code>npm run dev</code> in <code>frontend/</code>.
              </div>
            </div>

            <div
              className={`min-h-[160px] rounded-2xl border border-slate-800 bg-slate-950/60 p-4 text-sm text-slate-100 transition ${
                asrFlash ? "ring-2 ring-emerald-400/60" : "ring-1 ring-slate-800"
              }`}
            >
              <div className="flex items-center justify-between">
                <p className="text-xs uppercase tracking-[0.1em] text-slate-400">
                  Transcripts
                </p>
                {asrFlash && (
                  <span className="text-[11px] font-semibold text-emerald-300">
                    New ASR
                  </span>
                )}
              </div>
              {transcripts.length === 0 ? (
                <p className="mt-2 text-xs text-slate-400">
                  Utterances will appear here after VAD segmentation.
                </p>
              ) : (
                <div className="mt-3 space-y-2">
                  {transcripts.map((t, idx) => (
                    <div
                      key={idx}
                      className="rounded-xl bg-slate-900 px-3 py-2 text-sm text-slate-100 ring-1 ring-slate-800"
                    >
                      {t}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {toast && (
        <div className="fixed bottom-6 right-6 rounded-2xl bg-slate-900 px-4 py-3 text-sm text-slate-100 shadow-lg shadow-black/40 ring-1 ring-slate-700">
          {toast}
        </div>
      )}
    </div>
  );
}
