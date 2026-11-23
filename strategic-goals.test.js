import assert from 'node:assert/strict';
import { test } from 'node:test';
import { classify, createDeck, newGame, passTurn } from './engine.js';
import { legalFoundationActions } from './foundation-policy.js';
import { beliefInformation, sampleBeliefWorlds, summarizeBeliefs } from './strategic-beliefs.js';
import { chooseStrategicGoal } from './strategic-goals.js';

const deck = createDeck();
const own = [deck[3], deck[7], deck[13], deck[53]];
function endgame(hands) {
  const game = newGame(null, () => .42);
  const remaining = deck.filter(card => !hands.flat().includes(card) && card.id !== 4);
  game.hands = hands;
  game.played = hands.flatMap((hand, player) =>
    remaining.splice(0, 27 - hand.length - Number(player === 2)).map(card => ({ ...card, player })));
  game.played.push({ ...deck[4], player: 2 });
  game.order = hands.flatMap((hand, seat) => hand.length ? [] : [seat]);
  game.target = { cards: [deck[4]], type: classify([deck[4]], game.level) };
  game.lastPlayer = 2;
  return game;
}
const lastOpponent = id => endgame([own, [deck[id]], [], [], [], []]);

function decide(game) {
  const legal = legalFoundationActions(game);
  const scores = legal.map(action => action.kind === 'pass' ? .01 : 0);
  const result = chooseStrategicGoal(game, legal, scores);
  return { result, action: legal[result.index] };
}

test('exhaustive one-world endgame respects capacities, types and actual historical ask timing', () => {
  const game = lastOpponent(1);
  const info = beliefInformation(game);
  const higherIndex = game.played.findIndex(card => card.player === 1 && card.rank > 6);
  assert.ok(higherIndex !== -1);
  info.questions = [{ player: 1, playedAt: higherIndex, target: game.target.type }];
  const sampled = sampleBeliefWorlds(info, { samples: 4, attempts: 4 });
  assert.deepEqual(sampled.worlds[0].map(hand => hand.length), info.counts);
  assert.equal(new Set(sampled.worlds[0].flat().map(card => card.id)).size, 5);
  const beliefs = summarizeBeliefs(info, sampled, game.target.type);
  assert.equal(beliefs.expectedCounts[1][1], 1);
  assert.equal(beliefs.canOvertake[1], 0);
  assert.equal(beliefs.finishInOneIfLeading[1], 1);
  info.questions[0].playedAt = info.played.length;
  assert.equal(sampleBeliefWorlds(info, { samples: 4, attempts: 4 }).worlds.length, 0);
});

test('strategic belief inference uses no other actual hand or unrelated return type', () => {
  const game = newGame(null, () => .21);
  game.exchanges = [{ from: 2, to: 3, tribute: deck[53], returned: deck[18] }];
  const info = beliefInformation(game);
  assert.equal(info.exchanges[0].tributeRank, 20);
  assert.equal(info.exchanges[0].tributeType, null);
  assert.equal(info.exchanges[0].returnType, null);
  [game.hands[1], game.hands[4]] = [game.hands[4], game.hands[1]];
  game.exchanges[0].returned = deck[19];
  assert.deepEqual(beliefInformation(game), info);
  const draw = value => summarizeBeliefs(value, sampleBeliefWorlds(value, { samples: 4 }));
  assert.deepEqual(draw(beliefInformation(game)), draw(info));
});

test('an observed card and automatic highest-tribute rule constrain possible deals', () => {
  const game = lastOpponent(11);
  const info = beliefInformation(game);
  info.exchanges = [{ from: 1, to: 0, returned: true,
    tributeRank: 20, tributeType: 53, returnType: 11 }];
  assert.equal(sampleBeliefWorlds(info, { samples: 4, attempts: 4 }).worlds.length, 4);
  info.exchanges[0].returnType = 53;
  assert.equal(sampleBeliefWorlds(info, { samples: 4, attempts: 4 }).worlds.length, 0);
  info.exchanges[0] = { from: 1, to: 0, returned: true,
    tributeRank: 5, tributeType: 3, returnType: 11 };
  assert.equal(sampleBeliefWorlds(info, { samples: 4, attempts: 4 }).worlds.length, 0);
});

test('public pressure changes the goal with identical own cards, and only a legal action is returned', () => {
  const strong = decide(lastOpponent(11)), weak = decide(lastOpponent(1));
  assert.equal(strong.result.goal, 'defend');
  assert.equal(strong.result.currentOpportunity, 1);
  assert.equal(strong.result.selected.danger, 0);
  assert.equal(strong.action.kind, 'play');
  assert.equal(weak.result.goal, 'preserve');
  assert.equal(weak.action.kind, 'pass');
});

test('a previously passed opponent can still act before the trick closes', () => {
  const game = endgame([own, [], [], [], [deck[11], deck[12], deck[24], deck[25]], [deck[10]]]);
  game.passed = [5];
  const before = decide(game);
  assert.equal(before.result.goal, 'defend');
  assert.equal(before.result.currentOpportunity, 1);
  assert.equal(before.action.kind, 'play');
  [game.hands[4][0], game.hands[5][0]] = [game.hands[5][0], game.hands[4][0]];
  assert.deepEqual(decide(game), before);
  passTurn(game, 0);
  assert.equal(game.turn, 5);
  assert.ok(game.target);
});
