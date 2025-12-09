export async function playWavBytes(bytes: ArrayBuffer): Promise<void> {
  const blob = new Blob([bytes], { type: 'audio/wav' })
  const url = URL.createObjectURL(blob)
  const audio = new Audio(url)

  try {
    await audio.play()
  } finally {
    URL.revokeObjectURL(url)
  }
}
