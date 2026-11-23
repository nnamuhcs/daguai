import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir } from 'node:fs/promises';

const server = spawn(process.execPath, ['server.js'], { cwd: new URL('.', import.meta.url), env: { ...process.env, PORT: '0' }, stdio: ['ignore', 'pipe', 'pipe'] });
let serverError = '';
server.stderr.on('data', data => { serverError += data; });
let browser;
try {
  const url = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Server did not start: ${serverError}`)), 10000);
    server.once('error', error => { clearTimeout(timer); reject(error); });
    server.once('exit', code => { clearTimeout(timer); reject(new Error(`Server exited ${code}: ${serverError}`)); });
    server.stdout.on('data', data => {
      const match = String(data).match(/http:\/\/localhost:\d+/);
      if (match) { clearTimeout(timer); resolve(match[0]); }
    });
  });
  const response = await fetch(url);
  assert.equal(response.status, 200);
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const saved = () => page.evaluate(() => JSON.parse(localStorage.getItem('daguai-table-v2')));
  await page.goto(url);
  await page.locator('#hand .card').first().waitFor();
  assert.equal(await page.locator('.seat').count(), 6);
  assert.equal(await page.locator('#hand .card').count(), 27);
  assert.equal(await page.locator('.seat:not(.seat-0) .unreported').count(), 5);
  const handCards = page.locator('#hand .card');
  const firstCard = await handCards.nth(0).boundingBox();
  const fifthCard = await handCards.nth(4).boundingBox();
  await page.mouse.move(firstCard.x + 8, firstCard.y + 35);
  await page.mouse.down();
  await page.mouse.move(fifthCard.x + 8, fifthCard.y + 35, { steps: 12 });
  await page.mouse.up();
  assert.equal(await page.locator('#hand .selected').count(), 5);
  await page.mouse.move(firstCard.x + 8, firstCard.y + 35);
  await page.mouse.down();
  await page.mouse.move(fifthCard.x + 8, fifthCard.y + 35, { steps: 12 });
  await page.mouse.up();
  assert.equal(await page.locator('#hand .selected').count(), 0);
  await page.locator('#rules-button').click();
  assert.equal(await page.locator('#rules-dialog').evaluate(el => el.open), true);
  await page.locator('[data-close="rules-dialog"]').last().click();
  await page.locator('#hint-button').click();
  assert.ok(await page.locator('#hand .selected').count() > 0);
  assert.equal(await page.locator('#play-button').isEnabled(), true);
  const selected = await page.locator('#hand .selected').count();
  await page.locator('#play-button').click();
  assert.equal(await page.locator('#hand .card').count(), 27 - selected);
  const before = (await saved()).hands[0].map(c => c.id);
  await page.reload();
  assert.deepEqual((await saved()).hands[0].map(c => c.id), before);

  await page.evaluate(async () => {
    const { newGame, createDeck, playCards, sortHand } = await import('/engine.js');
    const game = newGame();
    const deck = createDeck();
    const players = [0, 5, 4, 3, 2, 1];
    const groups = players.map((_, i) => deck.filter(c => c.rank === 3 + i).slice(0, 5));
    const reserved = new Set(groups.flat().map(c => c.id));
    const rest = deck.filter(c => !reserved.has(c.id));
    players.forEach((player, i) => { game.hands[player] = sortHand([...groups[i], ...rest.slice(i * 22, i * 22 + 22)], game.level); });
    players.forEach((player, i) => playCards(game, player, groups[i].map(c => c.id)));
    localStorage.setItem('daguai-table-v2', JSON.stringify(game));
  });
  await page.reload();
  await page.locator('#pause-button').click();
  const tableSnapshot = await saved();
  assert.equal(await page.locator('.seat-cards .card').count(), 30);
  assert.match(await page.locator('.seat-1 .winning-play').getAttribute('aria-label'), /阿诚.*五同 8/);
  assert.match(await page.locator('#trick .trick-caption').textContent(), /阿诚.*五同 8/);
  assert.equal(await page.locator('#recent-plays .history-cards').count(), 6);
  const pausedState = JSON.stringify(await saved());
  await page.waitForTimeout(1500);
  assert.equal(JSON.stringify(await saved()), pausedState);
  await page.locator('#pass-button').click();
  assert.equal(await page.locator('.seat-0 .seat-cards .card').count(), 5);
  assert.match(await page.locator('.seat-0 .pass-label').textContent(), /不出/);

  await page.evaluate(async () => {
    const { newGame, playCards } = await import('/engine.js');
    const game = newGame();
    for (const player of [2, 4]) {
      const removed = game.hands[player].splice(1);
      game.played.push(...removed.map(c => ({ ...c, player })));
      game.turn = player;
      game.target = null;
      game.lastPlayer = null;
      playCards(game, player, [game.hands[player][0].id]);
    }
    localStorage.setItem('daguai-table-v2', JSON.stringify(game));
  });
  await page.reload();
  await page.locator('#pause-button').click();
  const finishedSnapshot = await saved();
  assert.equal(await page.locator('.seat.finished').count(), 2);
  assert.match(await page.locator('.seat-2 .finish-status').textContent(), /已出完.*头家/);
  assert.match(await page.locator('.seat-4 .finish-status').textContent(), /已出完.*第 2 名/);
  assert.equal(await page.locator('.seat.finished.active').count(), 0);
  assert.equal(await page.locator('#finish-strip .finish-chip').count(), 2);
  assert.ok((await page.locator('#scoreboard .team-row:not(.orange) .team-progress').textContent()).includes('已出完 2 / 3'));
  assert.equal(await page.locator('#scoreboard .member-finished').count(), 2);
  if (process.env.SCREENSHOT_DIR) {
    await mkdir(process.env.SCREENSHOT_DIR, { recursive: true });
    await page.screenshot({ path: `${process.env.SCREENSHOT_DIR}/daguai-finished-desktop.png`, fullPage: true });
  }

  await page.evaluate(async () => {
    const { newGame } = await import('/engine.js');
    const previous = newGame();
    previous.order = [0, 2, 4, 1, 3, 5];
    previous.result = { winner: 0, trapped: [5] };
    localStorage.setItem('daguai-table-v2', JSON.stringify(newGame(previous)));
  });
  await page.reload();
  assert.equal(await page.locator('#hand .card').count(), 28);
  assert.equal(await page.locator('#tribute-banner').isVisible(), true);
  assert.equal(await page.locator('#pass-button').isEnabled(), false);
  await page.locator('#hand .tribute').click({ position: { x: 8, y: 8 } });
  assert.equal(await page.locator('#play-button').isEnabled(), false);
  await page.locator('#hand .card:not(.tribute)').first().click({ position: { x: 8, y: 8 } });
  assert.equal(await page.locator('#hand .selected').count(), 1);
  assert.equal(await page.locator('#play-button').isEnabled(), true);
  const returnId = Number(await page.locator('#hand .selected').getAttribute('data-card'));
  await page.reload();
  assert.equal((await saved()).phase, 'returning');
  await page.locator(`[data-card="${returnId}"]`).click({ position: { x: 8, y: 8 } });
  await page.locator('#play-button').click();
  const returned = await saved();
  assert.equal(returned.phase, 'playing');
  assert.ok(returned.hands.every(h => h.length === 27));
  assert.equal(returned.exchanges[0].returned.id, returnId);
  assert.equal(await page.locator('#hand .card').count(), 27);

  await page.evaluate(async () => {
    const { newGame, playCards } = await import('/engine.js');
    const game = newGame();
    const owner = game.hands.findIndex((h, i) => i !== 0 && h.some(c => c.rank === 3));
    game.turn = owner;
    playCards(game, owner, [game.hands[owner].find(c => c.rank === 3).id]);
    game.turn = 0;
    localStorage.setItem('daguai-table-v2', JSON.stringify(game));
  });
  await page.reload();
  assert.equal(await page.locator('#ask-button').isEnabled(), true);
  await page.locator('#ask-button').click();
  assert.match(await page.locator('#toast').textContent(), /不报/);
  await page.evaluate(() => {
    const game = JSON.parse(localStorage.getItem('daguai-table-v2'));
    const owner = game.lastPlayer;
    const cards = game.hands[owner].splice(10);
    game.played.push(...cards.map(c => ({ ...c, player: owner })));
    game.reports[owner] = null;
    localStorage.setItem('daguai-table-v2', JSON.stringify(game));
  });
  await page.reload();
  await page.locator('#ask-button').click();
  assert.match(await page.locator('#toast').textContent(), /10 张/);
  const owner = (await saved()).lastPlayer;
  assert.match(await page.locator(`.seat-${owner} .seat-count`).textContent(), /10.*报数/);

  await page.locator('#restart-button').click();
  await page.locator('#confirm-restart').click();
  await page.locator('#speed-select').selectOption('250');
  await page.locator('#auto-toggle').check();
  await page.waitForFunction(() => JSON.parse(localStorage.getItem('daguai-table-v2'))?.over, null, { timeout: 120000 });
  assert.equal(await page.locator('#result-dialog').evaluate(el => el.open), true);
  assert.equal(await page.locator('.rank-item').count(), 6);
  const finished = await saved();
  assert.equal(finished.result.trapped.length, 6 - finished.result.finishedCount);
  assert.equal(await page.locator('.seat.finished').count(), finished.result.finishedCount);
  assert.equal(await page.locator('.seat.trapped').count(), finished.result.trapped.length);
  assert.equal(await page.locator('.seat.trapped .finish-status').count(), 0);
  assert.equal(await page.locator('#finish-strip .finish-chip').count(), finished.result.finishedCount);
  await page.locator('#next-button').click();
  await page.locator('#rules-button').click();
  assert.equal((await saved()).round, 2);
  const stable = JSON.stringify(await saved());
  await page.waitForTimeout(1000);
  assert.equal(JSON.stringify(await saved()), stable, 'modal must pause computer actions');
  await page.locator('[data-close="rules-dialog"]').last().click();

  const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  mobile.on('pageerror', error => errors.push(error.message));
  await mobile.goto(url);
  await mobile.locator('#hand .card').first().waitFor();
  assert.equal(await mobile.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await mobile.locator('#hint-button').click();
  assert.equal(await mobile.locator('#play-button').isEnabled(), true);
  await mobile.locator('#clear-button').click();
  await mobile.locator('#hand .card').first().scrollIntoViewIfNeeded();
  await mobile.locator('#hand').evaluate(el => { el.scrollLeft = 0; });
  const touchFirst = await mobile.locator('#hand .card').nth(0).boundingBox();
  const touchFourth = await mobile.locator('#hand .card').nth(3).boundingBox();
  const session = await mobile.context().newCDPSession(mobile);
  const point = (x, y) => ({ x, y, radiusX: 2, radiusY: 2 });
  await session.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [point(touchFirst.x + 8, touchFirst.y + 30)] });
  for (let step = 1; step <= 8; step++) {
    const x = touchFirst.x + 8 + (touchFourth.x - touchFirst.x) * step / 8;
    await session.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [point(x, touchFirst.y + 30)] });
  }
  await session.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  assert.equal(await mobile.locator('#hand .selected').count(), 4);
  await session.detach();
  await mobile.evaluate(game => localStorage.setItem('daguai-table-v2', JSON.stringify(game)), tableSnapshot);
  await mobile.reload();
  await mobile.locator('#pause-button').click();
  assert.equal(await mobile.locator('.seat-cards .card').count(), 30);
  assert.equal(await mobile.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  assert.ok(await mobile.evaluate(() => [...document.querySelectorAll('.seat-cards')].every(row =>
    [...row.children].every((card, index, cards) => index === 0 || card.getBoundingClientRect().left >= cards[index - 1].getBoundingClientRect().right))));
  await mobile.evaluate(game => localStorage.setItem('daguai-table-v2', JSON.stringify(game)), finishedSnapshot);
  await mobile.reload();
  await mobile.locator('#pause-button').click();
  assert.equal(await mobile.locator('.seat.finished').count(), 2);
  assert.match(await mobile.locator('#finish-strip').textContent(), /小满.*头家.*阿岚.*已出完/);
  assert.equal(await mobile.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  if (process.env.SCREENSHOT_DIR) {
    await mkdir(process.env.SCREENSHOT_DIR, { recursive: true });
    await mobile.screenshot({ path: `${process.env.SCREENSHOT_DIR}/daguai-mobile.png`, fullPage: true });
    await page.locator('#restart-button').click();
    await page.locator('#confirm-restart').click();
    await page.locator('#auto-toggle').uncheck();
    await page.screenshot({ path: `${process.env.SCREENSHOT_DIR}/daguai-desktop.png`, fullPage: true });
  }
  const neural = await browser.newPage();
  neural.on('pageerror', error => errors.push(error.message));
  await neural.route('**/api/ai/status', route => route.fulfill({ json: { available: true, games: 80 } }));
  let release;
  let requested;
  const scoring = new Promise(resolve => { requested = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  await neural.route('**/api/ai/score', async route => {
    const data = route.request().postDataJSON();
    assert.equal(data.observation.length, 922);
    assert.equal(data.actions[0].length, 71);
    assert.equal(data.hands, undefined);
    requested();
    await gate;
    await route.fulfill({ json: { scores: data.actions.map((_, i) => -i) } });
  });
  await neural.goto(url);
  await neural.waitForFunction(() => !document.querySelector('#ai-select [value="local"]').disabled);
  await neural.locator('#ai-select').selectOption('local');
  await neural.locator('#speed-select').selectOption('250');
  await neural.locator('#auto-toggle').check();
  await scoring;
  await neural.locator('#pause-button').click();
  release();
  await neural.waitForTimeout(500);
  assert.equal((await neural.evaluate(() => JSON.parse(localStorage.getItem('daguai-table-v2')))).history.length, 0);
  await neural.unroute('**/api/ai/score');
  await neural.route('**/api/ai/score', route => route.fulfill({ status: 503, json: { error: 'test model unavailable' } }));
  await neural.locator('#pause-button').click();
  await neural.waitForFunction(() => document.querySelector('#ai-select').value === 'heuristic');
  await neural.locator('#pause-button').click();
  assert.match(await neural.locator('#ai-status').textContent(), /模型失败.*规则电脑/);
  await neural.close();
  const fundamental = await browser.newPage();
  fundamental.on('pageerror', error => errors.push(error.message));
  await fundamental.route('**/api/ai/status', route => route.fulfill({
    json: { available: true, games: 32, encodingVersion: 2, learnReturns: true },
  }));
  await fundamental.route('**/api/ai/score', route => {
    const data = route.request().postDataJSON();
    assert.equal(data.actions[0].length, 81);
    assert.ok(data.history.every(token => token.length === 71));
    assert.equal(data.hands, undefined);
    const ask = data.actions.findIndex(action => action[74] === 1);
    return route.fulfill({ json: { scores: data.actions.map((_, i) => i === ask ? 100 : -i) } });
  });
  await fundamental.goto(url);
  await fundamental.waitForFunction(() => !document.querySelector('#ai-select [value="local"]').disabled);
  await fundamental.locator('#ai-select').selectOption('local');
  await fundamental.locator('#speed-select').selectOption('250');
  await fundamental.locator('#auto-toggle').check();
  await fundamental.waitForFunction(() => {
    const game = JSON.parse(localStorage.getItem('daguai-table-v2'));
    return game.history.length > 0 && game.foundationHistory.some(event => event.kind === 'ask');
  });
  await fundamental.locator('#pause-button').click();
  const foundationState = await fundamental.evaluate(() => JSON.parse(localStorage.getItem('daguai-table-v2')));
  const firstPlay = foundationState.foundationHistory.findIndex(event => event.kind === 'play');
  assert.deepEqual(foundationState.foundationHistory.slice(0, firstPlay).map(event => event.kind), ['ask', 'ask']);
  assert.equal(foundationState.policyEvents[0].pass, false);
  assert.match(await fundamental.locator('#ai-status').textContent(), /自主问牌/);
  await fundamental.close();
  assert.deepEqual(errors, []);
  console.log('Browser checks passed: prominent finished-player markers, ranks and team progress; six-player cards, history, pause, mouse/touch drag selection, tribute return, asking, saved resume, full round, next round, desktop/mobile.');
} finally {
  if (browser) await browser.close();
  server.kill('SIGTERM');
}
