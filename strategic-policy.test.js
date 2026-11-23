import assert from 'node:assert/strict';
import { test } from 'node:test';
import { classify, createDeck, newGame } from './engine.js';
import { legalFoundationActions } from './foundation-policy.js';
import { strategicFeatures } from './strategic-policy.js';

const deck = createDeck();
function position(ids) {
  const game = newGame(null, () => .42);
  game.hands[game.turn] = ids.map(id => deck[id]);
  return game;
}

test('own-hand cover and remainder are calculated per legal play, not by card count', () => {
  // Three independent pairs; a singleton strands its mate, a pair does not.
  const game = position([0, 13, 4, 17, 8, 21]);
  const legal = legalFoundationActions(game);
  const features = strategicFeatures(game, legal);
  const extra = predicate => features.actionExtras[legal.findIndex(predicate)];
  assert.equal(features.observation[3], 3 / 6);
  assert.equal(features.observation[4], 1);
  assert.equal(extra(action => action.kind === 'play' && action.ids.length === 2)[17], 2 / 6);
  assert.equal(extra(action => action.kind === 'play' && action.ids.length === 1)[17], 3 / 6);
  assert.equal(extra(action => action.kind === 'ask')[17], 3 / 6);
});

test('same own hand gets different tactical context under public danger and a teammate lead', () => {
  const game = position([8, 21, 10, 23]);
  game.target = { cards: [deck[1], deck[14]], type: classify([deck[1], deck[14]], game.level) };
  const withHolder = holder => {
    game.lastPlayer = holder;
    const legal = legalFoundationActions(game);
    return strategicFeatures(game, legal).actionExtras[legal.findIndex(action => action.kind === 'play')];
  };
  const ally = withHolder(2), opponent = withHolder(1);
  assert.deepEqual(ally.slice(10, 12), [1, 0]);
  assert.deepEqual(opponent.slice(10, 12), [0, 1]);
  game.played = deck.filter(card => !game.hands[0].includes(card)).slice(0, 26)
    .map(card => ({ ...card, player: 1 }));
  const urgent = withHolder(1);
  assert.equal(opponent[15], 1);
  assert.equal(urgent[15], 1 / 27);
});

test('only the acting seat remembers its own transfer; swapping hidden cards or copies is inert', () => {
  const game = position([0, 13, 8, 21]);
  game.exchanges = [
    { from: 1, to: 0, tribute: deck[52], returned: deck[51] },
    { from: 3, to: 2, tribute: deck[53], returned: deck[39] },
  ];
  const read = () => {
    const legal = legalFoundationActions(game);
    const encoded = strategicFeatures(game, legal);
    return {
      observation: encoded.observation,
      actions: legal.map((action, index) => ({
        identity: [action.kind, action.target, action.ids.map(id => id % 54).sort((a, b) => a - b)],
        extras: encoded.actionExtras[index],
      })).sort((a, b) => JSON.stringify(a.identity).localeCompare(JSON.stringify(b.identity))),
    };
  };
  const original = read();
  assert.equal(original.observation[8 + 52], 1);
  assert.equal(original.observation[8 + 61], 1);
  assert.equal(original.observation[8 + 62 + 51], 1);
  assert.equal(original.observation[8 + 62 + 60], 1);
  [game.hands[1], game.hands[3]] = [game.hands[3], game.hands[1]];
  game.exchanges[1].returned = deck[40];
  game.hands[0] = game.hands[0].map(card => deck[card.id + 54]);
  game.exchanges[0].tribute = deck[106];
  assert.deepEqual(read(), original);
});

test('next lead and a genuine team finish use the public table and correct direction', () => {
  const game = position([7]);
  game.order = [2, 4];
  const legal = legalFoundationActions(game);
  const [play] = strategicFeatures(game, legal).actionExtras;
  assert.deepEqual(play.slice(12, 14), [1, 1]);
  assert.equal(play[22], 0); // Seat 5 takes the next lead if the round continues.
});
