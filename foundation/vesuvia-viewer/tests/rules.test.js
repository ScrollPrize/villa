import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { join, relative, resolve, dirname } from 'node:path'

const ROOT = resolve(import.meta.dirname, '..')
const CORE = ['.', 'volume', 'gpu', 'views', 'overlays', 'ui']
const RANGED = 'volume/mirrors.js'
const BANNED_NAME = /\b\w*(Manager|Service|Handler|Impl|_v2|_fast)\b/
const HISTORY = /\b(previously|formerly|used to be|new in this)\b/i
const BANDAGE = /\b(workaround|hack|for now|temporary|temporarily|TODO|FIXME)\b/i
const RETRIED = ['aiohttp.ClientConnectorError']
const SPANNING = {
  'tests/consent.e2e.py': 'the question the core asks before a module loads',
}

const PRINTED = { ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓', '-': '−', ' ': 'Space' }
const UNLISTED = ['=']
const MODIFIERS = ['Shift']

const list = directory => readdirSync(join(ROOT, directory)).filter(name => name.endsWith('.js')).map(name => join(directory, name))
const read = file => readFileSync(join(ROOT, file), 'utf8')
const parts = readdirSync(join(ROOT, 'extensions'), { withFileTypes: true }).filter(entry => entry.isDirectory()).map(entry => join('extensions', entry.name))
const sources = [...CORE, 'extensions', ...parts].flatMap(list)
const tests = list('tests')
const checks = readdirSync(join(ROOT, 'tests')).filter(name => name.endsWith('.e2e.py')).map(name => join('tests', name))
const pages = readdirSync(join(ROOT, 'tests')).filter(name => name.endsWith('.e2e.js')).map(name => join('tests', name))
const tools = readdirSync(join(ROOT, 'tools')).filter(name => name.endsWith('.py')).map(name => join('tools', name))
const lines = file => read(file).split('\n').map((text, i) => ({ text, at: `${file}:${i + 1}` }))

test('a file exports only what another file or a test imports', () => {
  const imported = new Set()
  for (const file of [...sources, ...tests]) {
    for (const [, names, target] of read(file).matchAll(/\b(?:import|export)\s*\{([^}]*)\}\s*from\s*['"](\.[^'"]+)['"]/g)) {
      const from = relative(ROOT, resolve(ROOT, dirname(file), target))
      for (const name of names.split(',')) imported.add(`${from}:${name.trim().split(/\s+as\s+/)[0]}`)
    }
  }
  for (const file of sources) {
    const source = read(file)
    const exported = [...source.matchAll(/\bexport\s+(?:async\s+)?(?:function\*?|const|let|class)\s+(\w+)/g)]
      .map(match => ({ name: match[1], index: match.index }))
    for (const match of source.matchAll(/\bexport\s*\{([^}]*)\}/g)) {
      for (const name of match[1].split(',').map(name => name.trim()).filter(Boolean)) {
        exported.push({ name: name.split(/\s+as\s+/).at(-1), index: match.index })
      }
    }
    for (const { name, index } of exported) {
      if (name === 'default') continue
      assert.ok(imported.has(`${file}:${name}`), `${file}:${source.slice(0, index).split('\n').length}: exports ${name}, which nothing imports`)
    }
  }
})

test('a caught error is used: reported or thrown again, never dropped', () => {
  for (const file of sources) {
    const source = read(file)
    assert.ok(!/\bcatch\s*\{/.test(source), `${file} catches without naming the error`)
    for (const match of source.matchAll(/\bcatch\s*\((\w+)\)\s*\{/g)) {
      let depth = 1
      let end = match.index + match[0].length
      while (depth > 0) {
        if (source[end] === '{') depth += 1
        if (source[end] === '}') depth -= 1
        end += 1
      }
      const body = source.slice(match.index + match[0].length, end)
      const line = source.slice(0, match.index).split('\n').length
      assert.ok(new RegExp(`\\b${match[1]}\\b`).test(body), `${file}:${line} drops the error it catches`)
    }
  }
})

test('no base classes, and no names the rules ban', () => {
  for (const file of sources) {
    for (const { text, at } of lines(file)) {
      assert.ok(!/\bclass\s+\w+\s+extends\b/.test(text), `${at} extends a class`)
      assert.ok(!BANNED_NAME.test(text), `${at} uses the name ${text.match(BANNED_NAME)?.[0]}`)
    }
  }
})

test('documents and comments describe the present', () => {
  for (const file of ['README.md', ...sources, ...tools, ...checks]) {
    for (const { text, at } of lines(file)) {
      if (file.endsWith('.js') && !text.includes('//')) continue
      assert.ok(!HISTORY.test(text), `${at} tells history: ${text.trim()}`)
      assert.ok(!BANDAGE.test(text), `${at} describes what should be fixed: ${text.trim()}`)
    }
  }
})

test('a chunk is fetched and cached whole: only one file sends a Range header', () => {
  for (const file of sources.filter(name => name !== RANGED)) {
    for (const { text, at } of lines(file)) assert.ok(!/\bRange['"]?\s*:/.test(text), `${at} sends a Range header; fetch the whole chunk, or a large file or a TIFF tile through ${RANGED}`)
  }
})

test('a Python file holds no docstring', () => {
  for (const file of [...tools, ...checks]) {
    for (const { text, at } of lines(file)) assert.ok(!/^\s*"""/.test(text), `${at} is a docstring`)
  }
})

test('a Python tool takes argparse, and a caught error is used or named as retried', () => {
  const imported = new Set([...tools, ...checks].flatMap(file => [...read(file).matchAll(/^from (\w+) import/gm)].map(match => join('tools', `${match[1]}.py`))))
  for (const file of tools) {
    if (!imported.has(file)) assert.ok(read(file).includes('argparse.ArgumentParser('), `${file} takes no argparse`)
  }
  for (const file of [...tools, ...checks]) {
    const all = lines(file)
    all.forEach(({ text, at }, i) => {
      const clause = text.match(/^(\s*)except\b([^:]*):/)
      if (clause === null) return
      const named = clause[2].match(/\bas (\w+)$/)
      if (named === null) {
        assert.ok(RETRIED.includes(clause[2].trim()), `${at} drops the error it catches: ${text.trim()}`)
        return
      }
      const body = []
      for (let j = i + 1; j < all.length && (all[j].text.trim() === '' || all[j].text.match(/^\s*/)[0].length > clause[1].length); j++) body.push(all[j].text)
      assert.ok(body.some(line => new RegExp(`\\b${named[1]}\\b`).test(line)), `${at} drops the error it catches: ${text.trim()}`)
    })
  }
})

test("a core check's page loads no shipped extension", () => {
  for (const file of checks) {
    const name = file.slice('tests/'.length, -'.e2e.py'.length)
    if (file in SPANNING || readdirSync(join(ROOT, 'extensions')).includes(`${name}.js`)) continue
    assert.ok(!read(file).includes('ext=extensions/'), `${file} loads a shipped extension; its checks belong in tests/<extension>.e2e.py`)
  }
})

test('a browser check drives the real page: it replaces no browser API', () => {
  for (const file of [...checks, ...pages, 'tools/e2e.py']) {
    for (const { text, at } of lines(file)) {
      const replaced = text.match(/\b([A-Z]\w*(?:\.prototype)?\.\w+)\s*=[^=]/)
      assert.ok(replaced === null, `${at} replaces a browser API: ${text.trim()}`)
    }
  }
})

test('the help lists every key the core binds, and no other', () => {
  const keys = read('ui/keys.js')
  const actions = keys.slice(keys.indexOf('const actions = {'), keys.indexOf("document.addEventListener('keydown'"))
  const bound = [...actions.matchAll(/^ {4}(?:'([^']+)'|(\w+)):/gm)].map(match => match[1] ?? match[2]).filter(key => !UNLISTED.includes(key))
    .map(key => PRINTED[key] ?? key.replace(/^(Key|Digit)/, '').replace(/^[A-Z]$/, letter => letter.toLowerCase()))
  const html = read('index.html')
  const help = html.slice(html.indexOf('id="help-dialog"'))
  const listed = [...help.matchAll(/<kbd>([^<]+)<\/kbd>/g)].map(match => match[1]).filter(key => !MODIFIERS.includes(key))
  assert.deepEqual([...new Set(listed)].sort(), [...new Set(bound)].sort())
})
