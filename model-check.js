import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { chromium } from 'playwright';
import { newGame, computerTurn, validSave } from './engine.js';
import { observeForPolicy, legalPolicyActions } from './policy.js';
import { legalFoundationActions, encodeFoundationHistory, applyFoundationAction } from './foundation-policy.js';
import { strategicFeatures } from './strategic-policy.js';

const server = spawn(process.execPath, ['server.js'], {
  cwd: new URL('.', import.meta.url),
  env: { ...process.env, PORT: '0', AI_MODEL: process.env.AI_MODEL || 'training/checkpoints/latest.pt', AI_DEVICE: 'cpu' },
  stdio: ['ignore', 'pipe', 'pipe'],
});
let diagnostic = '';
server.stderr.on('data', data => { diagnostic += data; });
let browser;
const transport = { maxLegalActions: 0, maxBatchActions: 0, decisionLatencyMs: [] };
try {
  const url = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(diagnostic || 'Model test server did not start')), 15000);
    server.once('error', error => { clearTimeout(timer); reject(error); });
    server.once('exit', code => { clearTimeout(timer); reject(new Error(`Server exited ${code}: ${diagnostic}`)); });
    server.stdout.on('data', data => {
      const match = String(data).match(/http:\/\/localhost:\d+/);
      if (match) { clearTimeout(timer); resolve(match[0]); }
    });
  });
  const status = await (await fetch(`${url}/api/ai/status`)).json();
  const foundation = status.encodingVersion >= 2;
  const strategic = status.encodingVersion === 3;
  assert.equal(status.available, true, status.error);
  assert.ok(Number.isInteger(status.games) && status.games > 0);
  const reloaded = await (await fetch(`${url}/api/ai/reload`, { method: 'POST' })).json();
  assert.equal(reloaded.available, true);
  assert.equal(reloaded.games, status.games);
  assert.equal((await fetch(`${url}/api/ai/status`, { headers: { Origin: 'http://foreign.invalid' } })).status, 403);
  const invalid = await fetch(`${url}/api/ai/score`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ observation: [], actions: [[]] }),
  });
  assert.equal(invalid.status, 400);
  let game;
  for (let initial = 1; initial <= 32; initial++) {
    let seed = initial;
    const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
    const candidate = newGame(null, random);
    if ((foundation ? legalFoundationActions(candidate) : legalPolicyActions(candidate)).length > 512) {
      game = candidate;
      break;
    }
  }
  assert.ok(game, 'Need a deterministic multi-batch legal-support fixture');
  function featuresFor(current) {
    const actions = foundation ? legalFoundationActions(current) : legalPolicyActions(current);
    const extra = strategic ? strategicFeatures(current, actions) : null;
    if (extra) for (let i = 0; i < actions.length; i++)
      actions[i] = { ...actions[i], features: [...actions[i].features, ...extra.actionExtras[i]] };
    return { actions, observation: [...observeForPolicy(current), ...(extra?.observation ?? [])] };
  }
  async function choose(current) {
    const started = performance.now();
    const { actions, observation } = featuresFor(current);
    transport.maxLegalActions = Math.max(transport.maxLegalActions, actions.length);
    let best = -Infinity;
    let chosen;
    for (let offset = 0; offset < actions.length; offset += 512) {
      const batch = actions.slice(offset, offset + 512);
      transport.maxBatchActions = Math.max(transport.maxBatchActions, batch.length);
      const response = await fetch(`${url}/api/ai/score`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ observation, actions: batch.map(a => a.features),
          ...(foundation ? { history: encodeFoundationHistory(current) } : {}) }),
      });
      assert.equal(response.status, 200);
      const { scores } = await response.json();
      assert.equal(scores.length, batch.length);
      assert.ok(scores.every(Number.isFinite));
      scores.forEach((score, index) => {
        if (score > best) { best = score; chosen = batch[index]; }
      });
    }
    transport.decisionLatencyMs.push(performance.now() - started);
    return chosen;
  }
  for (let i = 0; i < 3 && !game.history.length; i++) {
    const chosen = await choose(game);
    if (foundation) applyFoundationAction(game, chosen);
    else computerTurn(game, chosen.ids);
  }
  assert.ok(validSave(game));
  assert.equal(game.history.at(-1).reason, '本地模型选择（试验）');
  const measuredDecisions = Number(process.env.MODEL_CHECK_DECISIONS || 0);
  assert.ok(Number.isInteger(measuredDecisions) && measuredDecisions >= 0 && measuredDecisions <= 500);
  for (let index = 0; index < measuredDecisions && !game.over; index++) {
    const chosen = await choose(game);
    if (foundation) applyFoundationAction(game, chosen);
    else computerTurn(game, chosen.ids);
    assert.ok(validSave(game));
  }
  const returning = newGame();
  const tribute = returning.hands[1].shift();
  returning.hands[0].push(tribute);
  returning.phase = 'returning';
  returning.exchanges = [{ to: 0, from: 1, tribute, returned: null }];
  const { actions: returnActions, observation: returnObservation } = featuresFor(returning);
  const returnResponse = await fetch(`${url}/api/ai/score`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ observation: returnObservation, actions: returnActions.map(a => a.features),
      ...(foundation ? { history: encodeFoundationHistory(returning) } : {}) }),
  });
  assert.equal(returnResponse.status, 200);
  const returnScores = (await returnResponse.json()).scores;
  const selectedReturn = returnScores.indexOf(Math.max(...returnScores));
  if (foundation) applyFoundationAction(returning, returnActions[selectedReturn]);
  else computerTurn(returning, returnActions[selectedReturn].ids);
  assert.ok(validSave(returning));
  assert.ok(returning.hands.every(hand => hand.length === 27));
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url);
  await page.waitForFunction(() => !document.querySelector('#ai-select [value="local"]').disabled, null, { timeout: 30000 });
  assert.equal(await page.locator('#ai-select').inputValue(), 'heuristic');
  await page.locator('#ai-select').selectOption('local');
  await page.locator('#speed-select').selectOption('250');
  await page.locator('#auto-toggle').check();
  await page.waitForFunction(() => JSON.parse(localStorage.getItem('daguai-table-v2')).history.some(h => h.reason === '本地模型选择（试验）'));
  await page.locator('#pause-button').click();
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('daguai-table-v2')));
  assert.ok(validSave(saved));
  await page.locator('#reload-model').click();
  await page.waitForFunction(() => !document.querySelector('#reload-model').disabled);
  assert.match(await page.locator('#ai-status').textContent(), new RegExp(`训练 ${status.games} 局`));
  if (status.learnReturns) {
    const returnGame = newGame();
    const tribute = returnGame.hands[1].shift();
    returnGame.hands[0].push(tribute);
    returnGame.phase = 'returning';
    returnGame.exchanges = [{ to: 0, from: 1, tribute, returned: null }];
    await page.evaluate(value => localStorage.setItem('daguai-table-v2', JSON.stringify(value)), returnGame);
    await page.reload();
    await page.waitForFunction(() => !document.querySelector('#ai-select [value="local"]').disabled);
    assert.match(await page.locator('#ai-status').textContent(), /含模型还贡/);
    await page.locator('#ai-select').selectOption('local');
    await page.locator('#auto-toggle').check();
    await page.waitForFunction(() => JSON.parse(localStorage.getItem('daguai-table-v2')).phase === 'playing');
    await page.locator('#pause-button').click();
    const returned = await page.evaluate(() => JSON.parse(localStorage.getItem('daguai-table-v2')));
    assert.notEqual(returned.exchanges[0].returned.id, tribute.id);
    assert.ok(validSave(returned));
  }
  console.log(`Local model checks passed: ${status.games} training games; real HTTP scores, engine validation, explicit browser selection and legal model play.`);
  if (process.env.MODEL_CHECK_REPORT) {
    const latencies = [...transport.decisionLatencyMs].sort((a, b) => a - b);
    const checkpoint = process.env.AI_MODEL || 'training/checkpoints/latest.pt';
    const report = {
      schemaVersion: 1, passed: true, sampling: 'greedy', allLegalActionsScored: true,
      checkpointSha256: createHash('sha256').update(await readFile(checkpoint)).digest('hex'),
      maxLegalActions: transport.maxLegalActions, maxBatchActions: transport.maxBatchActions,
      measuredDecisions: latencies.length,
      p95DecisionLatencyMs: latencies[Math.ceil(latencies.length * 0.95) - 1],
      scope: 'real localhost HTTP decisions plus explicit browser selection; not a strength claim',
    };
    const path = resolve(process.env.MODEL_CHECK_REPORT);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
  }
} finally {
  if (browser) await browser.close();
  server.kill('SIGTERM');
}
