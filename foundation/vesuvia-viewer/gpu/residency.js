import { WINDOW_CHUNKS, UPLOAD_BYTES_PER_FRAME } from '../config.js'
import { uniformLocations } from './gl.js'
import { MAX_LEVELS, PAGE_FILL, sampleGlsl } from './sample.glsl.js'

const PAGE_ABSENT = 0
const UNIFORMS = ['atlas', 'pages', 'levelCount', 'shape', 'chunk', 'downsample', 'offset', 'window', 'slots', 'slotSize', 'fill']

export class Residency {
  constructor(gl, volume, { atlasBytes, brickSize = null }) {
    // volume.levels with chunk_zyx replaced by the brick grid. The planner,
    // page tables and shader work in bricks; fetches and the CPU cache in source chunks.
    const levels = volume.levels.map(level => ({ ...level,
      chunk_zyx: level.chunk_zyx.map(s => brickSize !== null && s % brickSize === 0 ? brickSize : s),
    }))
    const maxTexture2D = gl.getParameter(gl.MAX_TEXTURE_SIZE)
    const maxTexture3D = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE)
    if (WINDOW_CHUNKS * WINDOW_CHUNKS > maxTexture2D) throw new Error(`page window height ${WINDOW_CHUNKS * WINDOW_CHUNKS} exceeds MAX_TEXTURE_SIZE ${maxTexture2D}`)
    if (levels.length > MAX_LEVELS) throw new Error(`${levels.length} levels; the page table holds at most ${MAX_LEVELS}`)
    const format = {
      uint8: { internal: gl.R8UI, format: gl.RED_INTEGER, type: gl.UNSIGNED_BYTE, bytes: 1 },
      uint16: { internal: gl.R16UI, format: gl.RED_INTEGER, type: gl.UNSIGNED_SHORT, bytes: 2 },
      float32: { internal: gl.R32F, format: gl.RED, type: gl.FLOAT, bytes: 4 },
    }[volume.dtype]

    const slotSize_zyx = [0, 1, 2].map(a => Math.max(...levels.map(level => level.chunk_zyx[a])))
    if (slotSize_zyx.some(s => s > maxTexture3D)) throw new Error(`chunk ${slotSize_zyx} exceeds MAX_3D_TEXTURE_SIZE ${maxTexture3D}`)
    if (slotSize_zyx[2] * format.bytes > UPLOAD_BYTES_PER_FRAME) throw new Error(`chunk row of ${slotSize_zyx[2] * format.bytes} bytes exceeds frame upload budget ${UPLOAD_BYTES_PER_FRAME}`)
    const slotBytes = slotSize_zyx[0] * slotSize_zyx[1] * slotSize_zyx[2] * format.bytes
    const slotBudget = Math.min(PAGE_FILL - 1, Math.floor(atlasBytes / slotBytes))
    if (slotBudget < 1) throw new Error(`atlas of ${atlasBytes} bytes holds no ${slotBytes}-byte slot`)
    const slotGrid_zyx = [1, 1, 1]
    for (;;) {
      const count = slotGrid_zyx[0] * slotGrid_zyx[1] * slotGrid_zyx[2]
      const axis = [0, 1, 2]
        .sort((a, b) => slotGrid_zyx[a] * slotSize_zyx[a] - slotGrid_zyx[b] * slotSize_zyx[b])
        .find(a => (slotGrid_zyx[a] + 1) * slotSize_zyx[a] <= maxTexture3D
          && count / slotGrid_zyx[a] * (slotGrid_zyx[a] + 1) <= slotBudget)
      if (axis === undefined) break
      slotGrid_zyx[axis] += 1
    }

    this.gl = gl
    this.volume = volume
    this.levels = levels
    this.format = format
    this.slotSize_zyx = slotSize_zyx
    this.slotGrid_zyx = slotGrid_zyx
    this.slotCount = slotGrid_zyx[0] * slotGrid_zyx[1] * slotGrid_zyx[2]
    this.brickBytes = levels.map(level => level.chunk_zyx.reduce((n, s) => n * s, format.bytes))

    this.atlas = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_3D, this.atlas)
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texStorage3D(gl.TEXTURE_3D, 1, format.internal,
      slotGrid_zyx[2] * slotSize_zyx[2], slotGrid_zyx[1] * slotSize_zyx[1], slotGrid_zyx[0] * slotSize_zyx[0])
    const refused = gl.getError()
    if (refused !== gl.NO_ERROR) throw new Error(`atlas of ${this.slotCount} slots of ${slotBytes} bytes: GL error 0x${refused.toString(16)}`)

    // Page layer layout: x = local cx, y = local cz * WINDOW_CHUNKS + local cy.
    this.pages = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.pages)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.R16UI, WINDOW_CHUNKS, WINDOW_CHUNKS * WINDOW_CHUNKS, levels.length)
    this.mirrors = levels.map(() => new Uint16Array(WINDOW_CHUNKS * WINDOW_CHUNKS * WINDOW_CHUNKS))
    this.windows = levels.map(() => null)
    this.pendingWindows = new Set()
    this.texel = new Uint16Array(1)

    this.slotByKey = new Map()
    this.freeSlots = Array.from({ length: this.slotCount }, (_, i) => this.slotCount - 1 - i)
    this.fills = new Map()
    this.stale = new Map()
    this.rewritten = levels.map(() => 0)
    this.protectedKeys = new Set()
    this.admission = []
    this.admissionByKey = new Map()
    this.upload = null
    this.residentBytes = 0
    this.plan = { wanted: [] }
    this.wantedByChunk = new Map()
    this.revision = 0
    this.counts = null

    const flat = pick => {
      const out = new Float32Array(3 * MAX_LEVELS)
      levels.forEach((level, i) => out.set(pick(level), 3 * i))
      return out
    }
    this.shape_zyx_L3 = flat(level => level.shape_zyx)
    this.chunk_zyx_L3 = flat(level => level.chunk_zyx)
    this.downsample_zyx_L3 = flat(level => level.downsample_zyx)
    this.offset_zyx_L3 = flat(level => level.offset_zyx)
    this.window_zyx_L3 = new Int32Array(3 * MAX_LEVELS)
    this.locations = new WeakMap()
  }

  windowIndex(level, cz, cy, cx) {
    const origin_zyx = this.windows[level]
    if (origin_zyx === null) return -1
    const lz = cz - origin_zyx[0]
    const ly = cy - origin_zyx[1]
    const lx = cx - origin_zyx[2]
    if ([lz, ly, lx].some(i => i < 0 || i >= WINDOW_CHUNKS)) return -1
    return (lz * WINDOW_CHUNKS + ly) * WINDOW_CHUNKS + lx
  }

  page(level, cz, cy, cx, entry, budget) {
    const index = this.windowIndex(level, cz, cy, cx)
    if (index === -1) return true
    const mirror = this.mirrors[level]
    if (mirror[index] === entry) return true
    if (budget.calls < 1 || budget.bytes < this.texel.byteLength) return false
    budget.calls--
    budget.bytes -= this.texel.byteLength
    mirror[index] = entry
    this.texel[0] = entry
    const gl = this.gl
    const lx = index % WINDOW_CHUNKS
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.pages)
    gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, lx, (index - lx) / WINDOW_CHUNKS, level, 1, 1, 1, gl.RED_INTEGER, gl.UNSIGNED_SHORT, this.texel)
    this.revision++
    return true
  }

  ensure(plan) {
    const volume = this.volume
    volume.clearQueue()
    this.plan = plan
    this.counts = null

    plan.windows.forEach((origin_zyx, level) => {
      const current = this.windows[level]
      if (origin_zyx === null || (current !== null && current.every((o, a) => o === origin_zyx[a]))) return
      this.windows[level] = origin_zyx
      const mirror = this.mirrors[level]
      mirror.fill(PAGE_ABSENT)
      for (const chunk of this.slotByKey.values()) {
        if (chunk.level !== level) continue
        const index = this.windowIndex(level, chunk.cz, chunk.cy, chunk.cx)
        if (index !== -1) mirror[index] = chunk.slot + 1
      }
      for (const chunk of this.fills.values()) {
        if (chunk.level !== level) continue
        const index = this.windowIndex(level, chunk.cz, chunk.cy, chunk.cx)
        if (index !== -1) mirror[index] = PAGE_FILL
      }
      this.pendingWindows.add(level)
    })

    this.protectedKeys = new Set(plan.wanted.map(c => volume.key(c.level, c.cz, c.cy, c.cx)))
    if (this.upload !== null && !this.protectedKeys.has(this.upload.key)) {
      if (!this.slotByKey.has(this.upload.key)) this.freeSlots.push(this.upload.slot)
      this.upload = null
    }
    this.admission = []
    this.admissionByKey = new Map()
    this.wantedByChunk = new Map()
    for (const c of plan.wanted) {
      const key = volume.key(c.level, c.cz, c.cy, c.cx)
      const brick_zyx = this.levels[c.level].chunk_zyx
      const chunk_zyx = volume.levels[c.level].chunk_zyx
      const parent_zyx = [c.cz, c.cy, c.cx].map((v, a) => Math.floor(v * brick_zyx[a] / chunk_zyx[a]))
      const parentKey = volume.key(c.level, ...parent_zyx)
      if (!this.wantedByChunk.has(parentKey)) this.wantedByChunk.set(parentKey, [])
      this.wantedByChunk.get(parentKey).push(c)
      if (this.upload !== null && this.upload.key === key) continue
      const resident = this.slotByKey.get(key)
      if (resident !== undefined) {
        this.slotByKey.delete(key)
        this.slotByKey.set(key, resident)
        if (!this.stale.has(key)) continue
      }
      if (this.fills.has(key)) continue
      const data = volume.cached(c.level, ...parent_zyx)
      if (data === undefined) {
        volume.request(c.level, ...parent_zyx, c.priority)
      } else if (data === null && this.pendingWindows.has(c.level)) {
        this.fills.set(key, { level: c.level, cz: c.cz, cy: c.cy, cx: c.cx })
        const index = this.windowIndex(c.level, c.cz, c.cy, c.cx)
        if (index !== -1) this.mirrors[c.level][index] = PAGE_FILL
      } else {
        const entry = { level: c.level, cz: c.cz, cy: c.cy, cx: c.cx, data }
        this.admission.push(entry)
        this.admissionByKey.set(key, entry)
      }
    }
    for (const c of plan.ahead) {
      const brick_zyx = this.levels[c.level].chunk_zyx
      const chunk_zyx = volume.levels[c.level].chunk_zyx
      volume.request(c.level, ...[c.cz, c.cy, c.cx].map((v, a) => Math.floor(v * brick_zyx[a] / chunk_zyx[a])), c.priority)
    }
  }

  // Windows go up before atlas mutations; eviction clears the page before slot reuse; a new brick is paged in only after every row is uploaded.
  tick(budget) {
    const gl = this.gl
    const volume = this.volume
    let changed = false
    for (const level of this.pendingWindows) {
      const mirror = this.mirrors[level]
      if (budget.calls < 1 || budget.bytes < mirror.byteLength) return changed
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.pages)
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, level, WINDOW_CHUNKS, WINDOW_CHUNKS * WINDOW_CHUNKS, 1,
        gl.RED_INTEGER, gl.UNSIGNED_SHORT, mirror)
      budget.calls--
      budget.bytes -= mirror.byteLength
      this.window_zyx_L3.set(this.windows[level], 3 * level)
      this.pendingWindows.delete(level)
      this.revision++
      changed = true
    }
    for (const chunk of volume.take()) {
      if (chunk.changed !== undefined) {
        const brick_zyx = this.levels[chunk.level].chunk_zyx
        const perChunk_zyx = volume.levels[chunk.level].chunk_zyx.map((s, a) => s / brick_zyx[a])
        for (let bz = 0; bz < perChunk_zyx[0]; bz++) for (let by = 0; by < perChunk_zyx[1]; by++) for (let bx = 0; bx < perChunk_zyx[2]; bx++) {
          const key = volume.key(chunk.level, chunk.cz * perChunk_zyx[0] + bz, chunk.cy * perChunk_zyx[1] + by, chunk.cx * perChunk_zyx[2] + bx)
          if (this.fills.delete(key)) this.counts = null
          const z0 = Math.max(0, chunk.changed[0] - bz * brick_zyx[0])
          const z1 = Math.min(brick_zyx[0], chunk.changed[1] - bz * brick_zyx[0])
          if (!this.slotByKey.has(key) || z0 >= z1) continue
          const owed = this.stale.get(key)
          this.stale.set(key, owed === undefined ? [z0, z1] : [Math.min(owed[0], z0), Math.max(owed[1], z1)])
        }
      }
      const bricks = this.wantedByChunk.get(volume.key(chunk.level, chunk.cz, chunk.cy, chunk.cx))
      if (bricks === undefined) continue
      for (const c of bricks) {
        const key = volume.key(c.level, c.cz, c.cy, c.cx)
        if (this.admissionByKey.has(key)) {
          this.admissionByKey.get(key).data = chunk.data
          continue
        }
        if (this.slotByKey.has(key) && !this.stale.has(key)) continue
        if (this.upload !== null && this.upload.key === key) {
          if (chunk.changed !== undefined) this.restart()
          continue
        }
        const entry = { level: c.level, cz: c.cz, cy: c.cy, cx: c.cx, data: chunk.data }
        this.admission.push(entry)
        this.admissionByKey.set(key, entry)
      }
    }

    while (budget.calls > 0 && budget.bytes > 0 && (this.upload !== null || this.admission.length > 0)) {
      if (this.upload === null) {
        const c = this.admission[0]
        const key = volume.key(c.level, c.cz, c.cy, c.cx)
        if (c.data === null) {
          if (!this.page(c.level, c.cz, c.cy, c.cx, PAGE_FILL, budget)) break
          this.fills.set(key, { level: c.level, cz: c.cz, cy: c.cy, cx: c.cx })
          this.admission.shift()
          this.admissionByKey.delete(key)
          changed = true
          continue
        }
        let slot
        if (this.slotByKey.has(key)) {
          slot = this.slotByKey.get(key).slot
        } else if (this.freeSlots.length > 0) {
          slot = this.freeSlots.pop()
        } else {
          let old
          for (const [oldKey, chunk] of this.slotByKey) {
            if (!this.protectedKeys.has(oldKey)) {
              old = chunk
              break
            }
          }
          if (old === undefined) break
          if (!this.page(old.level, old.cz, old.cy, old.cx, PAGE_ABSENT, budget)) break
          const oldKey = volume.key(old.level, old.cz, old.cy, old.cx)
          this.slotByKey.delete(oldKey)
          this.stale.delete(oldKey)
          slot = old.slot
          this.residentBytes -= this.brickBytes[old.level]
          changed = true
        }
        this.admission.shift()
        this.admissionByKey.delete(key)
        this.upload = { c, key, slot, z: 0, y: 0, zEnd: 0 }
        this.restart()
      }

      const { c, key, slot, z, y, zEnd } = this.upload
      const brick_zyx = this.levels[c.level].chunk_zyx
      const chunk_zyx = volume.levels[c.level].chunk_zyx
      if (z < zEnd) {
        const rowBytes = brick_zyx[2] * this.format.bytes
        const planeBytes = brick_zyx[1] * rowBytes
        if (budget.calls < 1 || budget.bytes < rowBytes) break
        const planes = y === 0 ? Math.min(zEnd - z, Math.floor(budget.bytes / planeBytes)) : 0
        const rows = planes > 0 ? brick_zyx[1] : Math.min(brick_zyx[1] - y, Math.floor(budget.bytes / rowBytes))
        const depth = Math.max(1, planes)
        const grid_zyx = this.slotGrid_zyx, size_zyx = this.slotSize_zyx
        const sz = Math.floor(slot / (grid_zyx[1] * grid_zyx[2]))
        const sy = Math.floor(slot / grid_zyx[2]) % grid_zyx[1]
        const sx = slot % grid_zyx[2]
        const offset = ((c.cz * brick_zyx[0] % chunk_zyx[0] + z) * chunk_zyx[1]
          + c.cy * brick_zyx[1] % chunk_zyx[1] + y) * chunk_zyx[2] + c.cx * brick_zyx[2] % chunk_zyx[2]
        gl.bindTexture(gl.TEXTURE_3D, this.atlas)
        gl.pixelStorei(gl.UNPACK_ROW_LENGTH, chunk_zyx[2])
        gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, chunk_zyx[1])
        gl.texSubImage3D(gl.TEXTURE_3D, 0, sx * size_zyx[2], sy * size_zyx[1] + y, sz * size_zyx[0] + z,
          brick_zyx[2], rows, depth, this.format.format, this.format.type, c.data, offset)
        gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0)
        gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, 0)
        budget.bytes -= depth * rows * rowBytes
        budget.calls--
        this.upload.y = (y + rows) % brick_zyx[1]
        this.upload.z += planes > 0 ? planes : (this.upload.y === 0 ? 1 : 0)
        if (this.slotByKey.has(key)) {
          this.rewritten[c.level]++
          changed = true
        }
        if (this.upload.z < zEnd) continue
      }
      if (!this.page(c.level, c.cz, c.cy, c.cx, slot + 1, budget)) break
      if (!this.slotByKey.has(key)) this.residentBytes += this.brickBytes[c.level]
      this.slotByKey.set(key, { slot, level: c.level, cz: c.cz, cy: c.cy, cx: c.cx })
      this.stale.delete(key)
      this.upload = null
      changed = true
    }
    return changed
  }

  restart() {
    const [z0, z1] = this.stale.get(this.upload.key) ?? [0, this.levels[this.upload.c.level].chunk_zyx[0]]
    this.upload.z = z0
    this.upload.y = 0
    this.upload.zEnd = z1
  }

  revisionAt(level) {
    let revision = this.revision
    for (let k = level; k < this.rewritten.length; k++) revision += this.rewritten[k]
    return revision
  }

  glsl(name) {
    return sampleGlsl(name, this.volume.dtype)
  }

  bind(program, name, unit) {
    const gl = this.gl
    let perProgram = this.locations.get(program)
    if (perProgram === undefined) {
      perProgram = {}
      this.locations.set(program, perProgram)
    }
    let loc = perProgram[name]
    if (loc === undefined) {
      const named = uniformLocations(gl, program, UNIFORMS.map(u => `${name}_${u}`))
      loc = {}
      for (const u of UNIFORMS) loc[u] = named[`${name}_${u}`]
      perProgram[name] = loc
    }
    gl.uniform1i(loc.levelCount, this.levels.length)
    gl.uniform3fv(loc.shape, this.shape_zyx_L3)
    gl.uniform3fv(loc.chunk, this.chunk_zyx_L3)
    gl.uniform3fv(loc.downsample, this.downsample_zyx_L3)
    gl.uniform3fv(loc.offset, this.offset_zyx_L3)
    gl.uniform3iv(loc.window, this.window_zyx_L3)
    gl.uniform3iv(loc.slots, this.slotGrid_zyx)
    gl.uniform3iv(loc.slotSize, this.slotSize_zyx)
    gl.uniform1f(loc.fill, this.volume.fill)
    gl.activeTexture(gl.TEXTURE0 + unit)
    gl.bindTexture(gl.TEXTURE_3D, this.atlas)
    gl.uniform1i(loc.atlas, unit)
    gl.activeTexture(gl.TEXTURE0 + unit + 1)
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.pages)
    gl.uniform1i(loc.pages, unit + 1)
    return unit + 2
  }

  close() {
    this.gl.deleteTexture(this.atlas)
    this.gl.deleteTexture(this.pages)
  }

  stats(view = null) {
    if (this.counts === null || this.counts.plan !== this.plan || this.counts.revision !== this.revision) {
      const counts = { plan: this.plan, revision: this.revision, wanted: this.plan.wanted.length, missing: 0, views: [] }
      for (const c of this.plan.wanted) {
        const key = this.volume.key(c.level, c.cz, c.cy, c.cx)
        if (this.slotByKey.has(key) || this.fills.has(key)) continue
        counts.missing++
        for (const v of c.views) counts.views[v] = (counts.views[v] ?? 0) + 1
      }
      this.counts = counts
    }
    return {
      residentBytes: this.residentBytes,
      slotsUsed: this.slotByKey.size,
      slotCount: this.slotCount,
      pending: this.pendingWindows.size + this.admission.length + (this.upload === null ? 0 : 1),
      wanted: this.counts.wanted,
      missing: view === null ? this.counts.missing : (this.counts.views[view] ?? 0),
    }
  }
}
