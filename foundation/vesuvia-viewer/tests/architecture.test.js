import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'

const ROOT = resolve(import.meta.dirname, '..')
const DIRECTORIES = ['.', 'volume', 'gpu', 'views', 'overlays', 'ui', 'extensions']
const imports = new Map()
const sourceFiles = directory => readdirSync(join(ROOT, directory), { withFileTypes: true }).flatMap(entry => {
  const file = join(directory, entry.name)
  if (entry.isDirectory()) return directory.startsWith('extensions') ? sourceFiles(file) : []
  return entry.name.endsWith('.js') ? [file] : []
})

function edges(source, file) {
  const found = []
  const add = (match, target) => found.push({ target, at: `${file}:${source.slice(0, match.index).split('\n').length}` })
  for (const match of source.matchAll(/\b(?:import|export)\s*(?:[\w$]+\s*,?\s*)?(?:\{[^}]*\}|\*(?:\s+as\s+\w+)?)?\s*from\s*(['"])([^'"]+)\1/g)) add(match, match[2])
  for (const match of source.matchAll(/\bimport\s*(['"])([^'"]+)\1/g)) add(match, match[2])
  for (const match of source.matchAll(/\bimport\s*\(\s*([^)]*)\)/g)) {
    const literal = /^(['"])([^'"]+)\1$/.exec(match[1].trim())
    if (literal !== null) add(match, literal[2])
    else assert.equal(file, 'main.js', `${file}:${source.slice(0, match.index).split('\n').length}: computed import ${match[1]} bypasses the dependency graph`)
  }
  for (const match of source.matchAll(/new\s+URL\s*\(\s*(['"])([^'"]+\.js)\1\s*,\s*import\.meta\.url\s*\)/g)) add(match, match[2])
  return found
}

for (const file of DIRECTORIES.flatMap(sourceFiles)) {
  imports.set(file, edges(readFileSync(join(ROOT, file), 'utf8'), file).map(({ target, at }) => {
    if (!file.startsWith('extensions/')) assert.ok(target.startsWith('.'), `${at}: core dependency ${target} must be a local file`)
    return { target: target.startsWith('.') ? relative(ROOT, resolve(ROOT, dirname(file), target)) : target, at }
  }))
}
const folder = file => file.includes('/') ? file.split('/')[0] : file

test('imports form no cycle', () => {
  const done = new Set()
  const visit = (file, path) => {
    assert.ok(!path.includes(file), `import cycle: ${[...path, file].join(' → ')}`)
    if (done.has(file)) return
    for (const { target } of imports.get(file) ?? []) visit(target, [...path, file])
    done.add(file)
  }
  for (const file of imports.keys()) visit(file, [])
})

test('folders import downwards only', () => {
  const allowed = {
    'main.js': ['volume', 'gpu', 'overlays', 'ui', 'views', 'vendor', 'scene.js', 'config.js'],
    'config.js': [],
    'scene.js': [],
    volume: ['volume', 'vendor', 'config.js'],
    gpu: ['gpu', 'config.js'],
    overlays: ['overlays', 'volume', 'vendor', 'config.js'],
    ui: ['ui', 'scene.js', 'config.js'],
    views: ['views', 'gpu', 'overlays', 'scene.js', 'config.js'],
    extensions: ['extensions', 'volume', 'gpu', 'overlays', 'scene.js', 'config.js'],
  }
  for (const [file, targets] of imports) {
    for (const { target, at } of targets) {
      if (folder(file) === 'extensions' && /^(https?:|[^./]+:)/.test(target)) continue
      assert.ok(allowed[folder(file)].includes(folder(target)), `${at}: must not import ${target}`)
    }
  }
})

test('an extension imports no other extension, only its own parts', () => {
  for (const [file, targets] of imports) {
    if (folder(file) !== 'extensions') continue
    const name = file.split('/')[1].replace(/\.js$/, '')
    for (const { target, at } of targets) {
      if (folder(target) !== 'extensions') continue
      assert.ok(target.startsWith(`extensions/${name}/`), `${at}: must not import ${target}`)
    }
  }
})

