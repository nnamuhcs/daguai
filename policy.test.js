import test from 'node:test';
import assert from 'node:assert/strict';
import { openEnvironment } from './training/test-bridge.js';
import { createDeck, newGame, classify, beats, playCards, returnCard, validSave } from './engine.js';
import { OBSERVATION_SIZE, ACTION_SIZE, observeForPolicy, legalPolicyActions, recordPolicyEvent } from './policy.js';

const shape = cards => cards.map(c => `${c.rank}:${c.suit}`).sort().join(',');
function exhaustive(hand, level, target = null) {
  const found = new Set();
  function choose(start, picked, count) {
    if (!count) {
      if (beats(classify(picked, level), target)) found.add(shape(picked));
      return;
    }
    for (let i = start; i <= hand.length - count; i++) choose(i + 1, [...picked, hand[i]], count - 1);
  }
  for (const count of [1, 2, 3, 5]) choose(0, [], count);
  return found;
}

test('complete policy actions match exhaustive physical-card enumeration including suits and wildcards', () => {
  const deck = createDeck();
  const fixtures = [
    [0, 1, 2, 3, 4, 54, 55, 56, 52, 53],
    [5, 18, 31, 44, 2, 15, 52, 106, 53],
    [52, 106, 160, 53, 107],
    [12, 0, 1, 2, 3, 39, 40, 41, 42, 43],
  ];
  let seed = 98;
  for (let n = 0; n < 12; n++) {
    const ids = new Set();
    while (ids.size < 10) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      ids.add(seed % 162);
    }
    fixtures.push([...ids]);
  }
  for (const ids of fixtures) for (const level of [2, 7]) {
    const game = newGame();
    game.hands[0] = ids.map(id => deck[id]);
    game.level = level;
    const actual = new Set(legalPolicyActions(game).map(action => {
      assert.equal(action.features.length, ACTION_SIZE);
      assert.ok(action.features.every(Number.isFinite));
      return shape(action.ids.map(id => deck[id]));
    }));
    assert.deepEqual(actual, exhaustive(game.hands[0], level), `${ids} level ${level}`);
    const targetCards = game.hands[0].slice(0, 1);
    game.target = { cards: targetCards, type: classify(targetCards, level) };
    const following = legalPolicyActions(game);
    assert.equal(following.at(-1).kind, 'pass');
    assert.deepEqual(new Set(following.filter(a => a.kind !== 'pass').map(a => shape(a.ids.map(id => deck[id])))),
      exhaustive(game.hands[0], level, game.target.type));
  }
});

test('policy observation contains only own cards and public facts, never hidden hands or returns', () => {
  const game = newGame();
  const before = observeForPolicy(game);
  assert.equal(before.length, OBSERVATION_SIZE);
  assert.ok(before.every(Number.isFinite));
  for (let seat = 1; seat < 6; seat++) game.hands[seat] = [];
  game.reports[2] = { answer: 5, playedAt: 0 };
  assert.deepEqual(observeForPolicy(game), before);
});

test('return actions can use another identical card but cannot return the received physical card', () => {
  const game = newGame();
  const deck = createDeck();
  game.phase = 'returning';
  game.hands[0] = [deck[53], deck[107], deck[2]];
  game.exchanges = [{ to: 0, from: 1, tribute: deck[53], returned: null }];
  const actions = legalPolicyActions(game);
  assert.ok(actions.every(a => a.kind === 'return'));
  assert.deepEqual(actions.map(a => a.ids[0]).sort((a, b) => a - b), [2, 107]);
});

test('outstanding tribute changes publicly known counts without exposing private returns', () => {
  const game = newGame();
  const tribute = game.hands[1].shift();
  game.hands[0].push(tribute);
  game.phase = 'returning';
  game.exchanges = [{ to: 0, from: 1, tribute, returned: null }];
  assert.ok(validSave(game));
  const observation = observeForPolicy(game);
  assert.equal(observation[378], 28 / 27);
  assert.equal(observation[379], 26 / 27);
  returnCard(game, 0, game.hands[0].find(c => c.id !== tribute.id).id);
  assert.ok(observeForPolicy(game).slice(378, 384).every(n => n === 1));
  assert.ok(validSave(game));
});

test('persisted policy history cannot introduce hidden cards into the observation', () => {
  const game = newGame();
  const action = legalPolicyActions(game)[0];
  playCards(game, 0, action.ids);
  recordPolicyEvent(game, 0, 0, 'playing');
  assert.ok(validSave(game));
  game.policyEvents[0].cards = [game.hands[1][0]];
  assert.equal(validSave(game), false);
});

test('persistent environment handles resets, full rounds, real tribute returns and shutdown', { timeout: 60000 }, async () => {
  const environment = openEnvironment('training/env.js');
  const { request } = environment;
  try {
    const info = await request({ op: 'hello' });
    assert.equal(info.observationSize, OBSERVATION_SIZE);
    const first = await request({ op: 'reset', seed: 123 });
    assert.deepEqual(await request({ op: 'reset', seed: 123 }), first);
    assert.ok((await request({ op: 'step', index: -1 })).error);
    const next = await request({ op: 'step', index: 0 });
    assert.equal(next.terminal, false);
    assert.equal(next.observation.length, OBSERVATION_SIZE);
    let finished;
    for (let seed = 0; seed < 8; seed++) {
      finished = await request({ op: 'reset', seed, level: 2 });
      for (let i = 0; !finished.terminal && i < 1500; i++) {
        finished = await request({ op: 'heuristic' });
        assert.equal(finished.error, undefined);
      }
      assert.equal(finished.terminal, true);
      assert.equal(finished.observation.length, OBSERVATION_SIZE);
      if (finished.result.winner !== null) break;
    }
    assert.notEqual(finished.result.winner, null);
    const returning = await request({ op: 'reset', seed: 100, continue: true });
    assert.equal(returning.phase, 'returning');
    assert.ok(returning.actions.every(action => action.kind === 'return'));
    const returned = await request({ op: 'step', index: 0 });
    assert.equal(returned.error, undefined);
    assert.equal(returned.observation.length, OBSERVATION_SIZE);
    assert.deepEqual(await request({ op: 'close' }), { closed: true });
  } finally {
    environment.close();
  }
});
