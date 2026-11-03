// Usage: node tools/vendor.mjs --modules <node_modules dir containing esbuild and numcodecs>
import { readFile, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'

const { values: { modules } } = parseArgs({ options: { modules: { type: 'string' } } })
if (modules === undefined) throw new Error('usage: node tools/vendor.mjs --modules <node_modules dir>')

const { build } = createRequire(`${modules}/`)('esbuild')
const outfile = fileURLToPath(new URL('../vendor/codecs.js', import.meta.url))

await build({
  stdin: {
    contents: "export { Blosc, Zstd, GZip, Zlib, LZ4 } from 'numcodecs'",
    resolveDir: modules,
    sourcefile: 'codecs-entry.js',
  },
  nodePaths: [modules],
  bundle: true,
  format: 'esm',
  legalComments: 'none',
  minify: true,
  outfile,
  platform: 'browser',
  target: 'es2022',
})

const licenses = []
for (const name of ['numcodecs', 'fflate']) {
  const { version } = JSON.parse(await readFile(`${modules}/${name}/package.json`, 'utf8'))
  const text = await readFile(`${modules}/${name}/LICENSE`, 'utf8')
  const heading = `${name} ${version}`
  licenses.push(`${heading}\n${'='.repeat(heading.length)}\n${text.trim()}`)
}
await writeFile(new URL('../vendor/LICENSES.txt', import.meta.url), `${licenses.join('\n\n')}\n`)
console.log(`wrote ${outfile}`)
