// Held keys run off one rAF loop: the browser's own autorepeat starts too late and fires too coarsely.
import { ZOOM_STEP, HELD_ZOOM_OCTAVE_MS, HELD_STEP_MS, HELD_ROTATE_MS, HELD_WATCHDOG_MS, ROTATE_STEP_RADIANS } from '../config.js'

const STEPS = { KeyD: -1, KeyF: 1 }
const ZOOMS = { KeyI: 1, KeyO: -1 }
const TURNS = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }

const glide = (rate, tap, move) => ({ rate, tap, move, direction: null, down: false, left: 0, time: 0 })

export function heldKeys({ step, zoom, turn }) {
  const zoomGlide = glide(1 / HELD_ZOOM_OCTAVE_MS, Math.log2(ZOOM_STEP), (direction, octaves) => zoom(2 ** (-direction * octaves)))
  const turnGlide = glide(ROTATE_STEP_RADIANS / HELD_ROTATE_MS, ROTATE_STEP_RADIANS, ([right, down], radians) => turn(right * radians, down * radians))
  const held = { step: 0, stepTime: 0, frame: null, watchdog: null }

  const advance = (motion, time) => {
    if (motion.direction === null) return
    const amount = Math.min(Math.max(0, time - motion.time) * motion.rate, motion.down ? Infinity : motion.left)
    motion.time = time
    motion.left -= amount
    motion.move(motion.direction, amount)
    if (!motion.down && motion.left <= 0) motion.direction = null
  }
  const heldFrame = time => {
    held.frame = null
    if (held.step !== 0 && time - held.stepTime >= HELD_STEP_MS) {
      held.stepTime = time
      step(held.step)
    }
    advance(zoomGlide, time)
    advance(turnGlide, time)
    if (held.step !== 0 || zoomGlide.direction !== null || turnGlide.direction !== null) held.frame = requestAnimationFrame(heldFrame)
  }
  const stop = () => {
    held.step = 0
    zoomGlide.direction = turnGlide.direction = null
  }
  const arm = () => {
    clearTimeout(held.watchdog)
    held.watchdog = setTimeout(stop, HELD_WATCHDOG_MS)
    if (held.frame === null) held.frame = requestAnimationFrame(heldFrame)
  }
  const holdStep = (direction, repeat) => {
    if (!repeat) {
      step(direction)
      held.stepTime = performance.now()
    }
    held.step = direction
    arm()
  }
  const hold = (motion, direction, repeat) => {
    if (motion.direction !== direction) {
      motion.time = performance.now()
      motion.left = 0
    }
    if (!repeat) motion.left += motion.tap
    motion.direction = direction
    motion.down = true
    arm()
  }
  const release = (motion, direction) => {
    if (motion.direction !== direction) return
    motion.down = false
    if (motion.left <= 0) motion.direction = null
  }

  return {
    stop,
    press(key, repeat) {
      if (key in STEPS) holdStep(STEPS[key], repeat)
      if (key in ZOOMS) hold(zoomGlide, ZOOMS[key], repeat)
      if (key in TURNS) hold(turnGlide, TURNS[key], repeat)
    },
    release(key) {
      if (key in STEPS) held.step = 0
      if (key in ZOOMS) release(zoomGlide, ZOOMS[key])
      if (key in TURNS) release(turnGlide, TURNS[key])
    },
  }
}
