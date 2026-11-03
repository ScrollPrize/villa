export class ChunkCache {
  constructor(budgetBytes) {
    this.budgetBytes = budgetBytes
    this.bytes = 0
    this.lru = new Map()
    this.fill = new Set()
  }

  has(key) {
    return this.lru.has(key) || this.fill.has(key)
  }

  get(key) {
    if (this.fill.has(key)) return null
    const data = this.lru.get(key)
    if (data === undefined) return undefined
    this.lru.delete(key)
    this.lru.set(key, data)
    return data
  }

  set(key, data) {
    if (data === null) {
      this.fill.add(key)
      return
    }
    const old = this.lru.get(key)
    if (old !== undefined) {
      this.bytes -= old.byteLength
      this.lru.delete(key)
    }
    this.lru.set(key, data)
    this.bytes += data.byteLength
    for (const [oldKey, oldData] of this.lru) {
      if (this.bytes <= this.budgetBytes) break
      this.lru.delete(oldKey)
      this.bytes -= oldData.byteLength
    }
  }
}
