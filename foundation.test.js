import test from 'node:test';
import assert from 'node:assert/strict';
import { newGame, playCards, passTurn, publicPolicyHistory, validSave, computerTurn } from './engine.js';
import { legalPolicyActions } from './policy.js';
import { legalFoundationActions, encodeFoundationHistory, applyFoundationAction } from './foundation-policy.js';
import { lateGame, leadFor, tacticalProbes } from './training/foundation-scenarios.js';
import { openEnvironment } from './training/test-bridge.js';

test('asking is a chosen legal action, reveals only the public answer, and cannot loop unchanged queries', () => {
  const game = lateGame([2, 15, 5, 18, 12, 25]);
  leadFor(game, 2, [55, 68]);
  passTurn(game, 1);
  const before = game.hands.map(hand => hand.map(c => c.id));
  const ask = legalFoundationActions(game).find(action => action.kind === 'ask');
  assert.equal(ask.target, 2);
  assert.equal(ask.features.length, 81);
  applyFoundationAction(game, ask);
  assert.equal(game.turn, 0);
  assert.deepEqual(game.hands.map(hand => hand.map(c => c.id)), before);
  assert.ok(!legalFoundationActions(game).some(action => action.kind === 'ask'));
  const token = encodeFoundationHistory(game).at(-1);
  assert.equal(token.length, 71);
  assert.ok(token.slice(0, 54).every(value => value === 0));
  assert.equal(token[54], 1);
  assert.equal(token[62], 1);
  assert.equal(token[68], 1);
  assert.equal(token[69], -1);
  assert.ok(validSave(game));
  const unchanged = JSON.stringify(game);
  assert.throws(() => applyFoundationAction(game, null), /不合法/);
  assert.throws(() => applyFoundationAction(game, ask), /不合法/);
  assert.equal(JSON.stringify(game), unchanged);
});

test('foundation model card choices do not trigger rule-based automatic asking', () => {
  const game = newGame();
  const action = legalFoundationActions(game).find(a => a.kind === 'play');
  applyFoundationAction(game, action);
  assert.equal(game.questions.length, 0);
  assert.deepEqual(game.foundationHistory.map(event => event.kind), ['play']);
  const old = newGame();
  const move = legalPolicyActions(old)[0];
  computerTurn(old, move.ids);
  assert.ok(validSave(old));
});

test('whole observed public history survives tricks and save restoration without revealing hidden cards', () => {
  const game = newGame();
  const move = legalPolicyActions(game)[0];
  playCards(game, 0, move.ids);
  for (const player of [5, 4, 3, 2, 1]) passTurn(game, player);
  assert.deepEqual(publicPolicyHistory(game).map(event => event.kind), ['play', 'pass', 'pass', 'pass', 'pass', 'pass']);
  assert.equal(game.target, null);
  const before = encodeFoundationHistory(game);
  const restored = JSON.parse(JSON.stringify(game));
  assert.ok(validSave(restored));
  assert.deepEqual(encodeFoundationHistory(restored), before);
  [restored.hands[1], restored.hands[3]] = [restored.hands[3], restored.hands[1]];
  assert.deepEqual(encodeFoundationHistory(restored), before);
  const hidden = restored.hands[1][0];
  restored.foundationHistory.push({ kind: 'play', player: 1, cards: [hidden], lead: false });
  assert.equal(validSave(restored), false);
});

test('legacy public history remains available and no private return cards become event tokens', () => {
  const game = newGame();
  playCards(game, 0, legalPolicyActions(game)[0].ids);
  delete game.foundationHistory;
  assert.equal(publicPolicyHistory(game).length, 1);
  const tampered = JSON.parse(JSON.stringify(game));
  tampered.history.push({ player: 1, cards: [tampered.hands[1][0]], lead: false });
  assert.equal(validSave(tampered), false);
  passTurn(game, 5);
  assert.deepEqual(publicPolicyHistory(game).map(event => event.kind), ['play', 'pass']);
  const returning = newGame();
  const tribute = returning.hands[1].shift();
  returning.hands[0].push(tribute);
  returning.phase = 'returning';
  returning.exchanges = [{ to: 0, from: 1, tribute, returned: null }];
  applyFoundationAction(returning, legalFoundationActions(returning)[0]);
  assert.deepEqual(encodeFoundationHistory(returning), []);
  assert.ok(validSave(returning));
});

test('tactical reference diagnostics cover feed and Joker failures using legal physical cards', () => {
  const probes = tacticalProbes();
  assert.equal(probes.length, 7);
  for (const probe of probes) {
    assert.equal(probe.observation.length, 922);
    assert.ok(probe.actions.some(action => action.expected));
    assert.ok(probe.actions.every(action => action.features.length === 81 && action.kind !== 'ask'));
    assert.ok(probe.history.every(event => event.length === 71));
  }
});

test('foundation bridge emits only new public events, learns asking, resets history and supports real returns', { timeout: 60000 }, async () => {
  const environment = openEnvironment('training/foundation-env.js');
  const { request } = environment;
  try {
    assert.deepEqual(await request({ op: 'hello' }), { encodingVersion: 2, observationSize: 922, actionSize: 81, historySize: 71 });
    const first = await request({ op: 'reset', seed: 0, startingSeat: 0, level: 2 });
    assert.deepEqual(first.events, []);
    const ask = first.actions.findIndex(action => action.kind === 'ask');
    const asked = await request({ op: 'step', index: ask });
    assert.equal(asked.player, first.player);
    assert.equal(asked.events.length, 1);
    assert.equal(asked.events[0][68], 1);
    assert.ok(!asked.actions.some(action => action.kind === 'ask' && action.target === first.actions[ask].target));
    const played = await request({ op: 'step', index: asked.actions.findIndex(action => action.kind === 'play') });
    assert.equal(played.events.length, 1);
    assert.equal(played.events[0][66], 1);
    assert.deepEqual(await request({ op: 'reset', seed: 0, startingSeat: 0, level: 2 }), first);
    let finished;
    for (let seed = 0; seed < 5; seed++) {
      finished = await request({ op: 'reset', seed, level: 2 });
      for (let i = 0; !finished.terminal && i < 6000; i++) {
        finished = await request({ op: 'heuristic' });
        assert.equal(finished.error, undefined);
        assert.ok(finished.events.every(token => token.length === 71));
      }
      assert.equal(finished.terminal, true);
      if (finished.result.winner !== null) break;
    }
    assert.notEqual(finished.result.winner, null);
    const returning = await request({ op: 'reset', seed: 100, continue: true });
    assert.equal(returning.phase, 'returning');
    assert.deepEqual(returning.events, []);
    assert.ok(returning.actions.every(action => action.kind === 'return'));
    const returned = await request({ op: 'step', index: 0 });
    assert.equal(returned.error, undefined);
    assert.deepEqual(returned.events, []);
    assert.deepEqual(await request({ op: 'close' }), { closed: true });
  } finally {
    environment.close();
  }
});
