// Pair browser behavior, with controlled DOM/fetch and no network or models.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = fs.readFileSync(path.join(__dirname, '../app/static/pairs.js'), 'utf8');

function env({ empty = false } = {}) {
  const elements = new Map(), calls = [];
  function element() {
    return { hidden: false, disabled: false, value: '', textContent: '', options: [], handlers: {},
      addEventListener(event, fn) { this.handlers[event] = fn; },
      appendChild(option) { this.options.push(option); if (!this.value) this.value = option.value; } };
  }
  function get(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }
  const context = vm.createContext({
    document: { getElementById: get, createElement: element }, encodeURIComponent,
    fetch: async (url, options) => {
      calls.push({ url, options });
      const data = url.endsWith('/runs') ? { runs: empty ? [] : [{ run_id: 'sweep-test', pairs: 14 }] } : url.endsWith('/manifest') ?
        { run_id: 'sweep-test', pairs: Array.from({ length: 14 }, (_, i) => ({ pair_id: `pair-${String(i + 1).padStart(5, '0')}` })) } :
        { imported: 1, kind: 'model', summary: { honesty: 'Model labels are model-read; human spot-check 0 of 1.' } };
      return { ok: true, json: async () => data };
    }
  });
  vm.runInContext(script, context);
  return { get, calls, context, click: async (id) => get(id).handlers.click() };
}

async function main() {
  const e = env();
  await e.click('pairView');
  assert.equal(e.get('labelRoot').hidden, true);
  assert.equal(e.get('pairRoot').hidden, false);
  assert.equal(e.get('pairImage').src, '/api/pairs/sweep-test/sheet/pair-00001');
  assert.equal(e.get('pairDownload').href, e.get('pairImage').src);
  assert.equal(e.get('pairPrev').disabled, true);
  for (let i = 0; i < 12; i++) await e.click('pairNext');
  assert.equal(e.get('pairContact').href, '/api/pairs/sweep-test/contact/2');
  await e.click('pairNext');
  await e.click('pairNext');
  assert.equal(e.get('pairImage').src, '/api/pairs/sweep-test/sheet/pair-00014');
  assert.equal(e.get('pairNext').disabled, true);
  await e.click('pairPrev');
  assert.match(e.get('pairPosition').textContent, /13 \/ 14/);
  e.get('pairAnnotator').value = 'chatgpt-example';
  e.get('pairKind').value = 'model';
  e.get('pairAnswers').value = '{"pair_id":"pair-00013","answer":"same","reason":"Matching wear."}';
  await e.click('pairImport');
  const sent = e.calls.at(-1);
  assert.equal(sent.url, '/api/pairs/sweep-test/import');
  assert.equal(JSON.parse(sent.options.body).kind, 'model');
  assert.match(e.get('pairMsg').textContent, /model-read; human spot-check 0 of 1/);
  assert.equal(e.get('pairAnswers').value, '');
  assert(!e.calls.some(({ url }) => url.endsWith('/summary') || url.includes('/key')));
  await e.click('pairTitles');
  assert.equal(e.get('labelRoot').hidden, false);
  assert.equal(e.get('pairRoot').hidden, true);
  const empty = env({ empty: true });
  await empty.click('pairView');
  assert.match(empty.get('pairMsg').textContent, /No pair sheets/);
  await empty.click('pairImport');
  assert.match(empty.get('pairMsg').textContent, /Load a sheet set first/);
  const error = env();
  await error.click('pairView');
  error.get('pairAnswers').value = 'answers to preserve';
  error.context.fetch = async () => ({ ok: false, json: async () => ({ detail: 'invalid answer' }) });
  await error.click('pairImport');
  assert.equal(error.get('pairMsg').textContent, 'Not imported: invalid answer');
  assert.equal(error.get('pairAnswers').value, 'answers to preserve');
  console.log('Pair client regressions passed: navigation, blinding, provenance, empty/error states.');
}
main().catch((e) => { console.error(e); process.exitCode = 1; });
