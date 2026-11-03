export function openSurface(url, { maxPoints }) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./tifxyz-worker.js', import.meta.url), { type: 'module' })
    worker.onmessage = ({ data }) => {
      worker.terminate()
      if (data.error !== undefined) reject(new Error(data.error))
      else resolve(data)
    }
    worker.onerror = event => {
      event.preventDefault()
      worker.terminate()
      reject(new Error(event.message))
    }
    worker.postMessage({ url: new URL(url, location.href).href, maxPoints })
  })
}
