export function element(id) {
  const found = document.getElementById(id)
  if (found === null) throw new Error(`index.html has no #${id}`)
  return found
}
