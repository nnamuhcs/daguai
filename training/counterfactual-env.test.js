import test from 'node:test';
import assert from 'node:assert/strict';
import { CounterfactualEnvironment, instantiateBeliefWorld, checkAuditBudget } from './counterfactual-env.js';
import { beliefInformation, sampleBeliefWorlds } from '../strategic-beliefs.js';
import { validSave } from '../engine.js';

test('audit uses legal moves and does not mutate the original position', () => {
  const env = new CounterfactualEnvironment();
  const state = env.handle({ op: 'sample', seed: 20850001, phase: 'mid' });
  const initial = JSON.stringify(env.game);
  const result = env.handle({ op: 'counterfactual_audit', indices: [0, 1], worlds: 4, attempts: 64, seed: 79 });
  assert.equal(result.status, 'ok');
  assert.equal(result.sampledWorlds, 4);
  assert.equal(result.evaluatedRollouts, 8);
  assert.equal(JSON.stringify(env.game), initial);
  assert.ok(result.estimates.every(x => x.score >= 0 && x.score <= 1));
  assert.equal(result.estimates[0].pairedDeltaVsChosen, 0);
  assert.deepEqual(result.estimates[0].paired95Approx, [0, 0]);
  assert.equal(state.actions.length, env.actions.length);
  assert.ok(!('hiddenHands' in result));
});

test('possible worlds do not depend on real opponents private cards', () => {
  const env = new CounterfactualEnvironment();
  env.handle({ op: 'sample', seed: 20850017, phase: 'fresh' });
  const state = structuredClone(env.game);
  const info = beliefInformation(state);
  const worlds = sampleBeliefWorlds(info, { samples: 4, attempts: 64, seed: 799 }).worlds;
  assert.equal(worlds.length, 4);
  const replaced = instantiateBeliefWorld(state, worlds[0]);
  assert.ok(validSave(replaced));
  assert.deepEqual(replaced.hands[state.turn], state.hands[state.turn]);
  assert.deepEqual(replaced.played, state.played);
  // A different hidden state with the same own/public facts must yield the same audit result.
  env.game = replaced;
  env.state();
  const result1 = env.handle({ op: 'counterfactual_audit', indices: [0, 1], worlds: 4, attempts: 64, seed: 797 });
  env.game = state;
  env.state();
  const result2 = env.handle({ op: 'counterfactual_audit', indices: [0, 1], worlds: 4, attempts: 64, seed: 797 });
  assert.deepEqual(result1, result2);
});

test('sampled tribute exchanges use possible-world cards and preserve the legal return', () => {
  const env = new CounterfactualEnvironment();
  const state = env.handle({op: 'sample', seed: 20853412, phase: 'returning'});
  const game = env.game;
  assert.ok(game.exchanges.length > 1);
  const worlds = sampleBeliefWorlds(beliefInformation(game), {samples: 4, attempts: 1024, seed: 20853413}).worlds;
  assert.equal(worlds.length, 4);
  for (const hands of worlds) {
    const world = instantiateBeliefWorld(game, hands);
    for (const {to, tribute} of world.exchanges) {
      assert.ok(world.hands[to].some(card => card.id === tribute.id));
    }
    assert.equal(world.exchanges[world.returnIndex].tribute.id, game.exchanges[game.returnIndex].tribute.id);
  }
  assert.equal(state.actions[0].kind, 'return');
});

test('budget rejects unbounded requests', () => {
  assert.throws(() => checkAuditBudget({indices: [0, 1], worlds: 3, attempts: 64, seed: 1}), /budget/);
  assert.throws(() => checkAuditBudget({indices: [0, 1], worlds: 32, attempts: 1, seed: 1}), /budget/);
  assert.throws(() => checkAuditBudget({indices: [0, 0], worlds: 4, attempts: 4, seed: 1}), /budget/);
  assert.throws(() => checkAuditBudget({indices: Array.from({length: 12}, (_, i) => i), worlds: 32, attempts: 32, seed: 1}), /budget/);
});
