import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { newGame, createDeck, playCards, passTurn, validSave, rankName, classify } from '../engine.js';
import { observeFoundation, legalFoundationActions, encodeFoundationHistory } from '../foundation-policy.js';

export function lateGame(ids) {
  const game = newGame();
  const deck = createDeck();
  const own = ids.map(id => deck[id]);
  const other = deck.filter(c => !ids.includes(c.id));
  const used = other.splice(0, 27 - own.length);
  game.hands = [own, ...Array.from({ length: 5 }, (_, i) => other.slice(i * 27, (i + 1) * 27))];
  game.played = used.map(c => ({ ...c, player: 0 }));
  assert.ok(validSave(game));
  return game;
}

export function leadFor(game, player, ids) {
  for (const id of ids) {
    const owner = game.hands.findIndex(h => h.some(c => c.id === id));
    if (owner === player) continue;
    const from = game.hands[owner].findIndex(c => c.id === id);
    const to = game.hands[player].findIndex(c => !ids.includes(c.id));
    [game.hands[owner][from], game.hands[player][to]] = [game.hands[player][to], game.hands[owner][from]];
  }
  game.turn = player;
  playCards(game, player, ids);
  assert.ok(validSave(game));
}

export function tacticalProbes() {
  const probes = [];
  function add(name, game, expected) {
    const key = [...expected].sort((a, b) => a - b).join(',');
    const actions = legalFoundationActions(game).filter(a => a.kind !== 'ask').map(action => {
      const cards = action.ids.map(id => game.hands[game.turn].find(c => c.id === id));
      return {
        ...action, expected: [...action.ids].sort((a, b) => a - b).join(',') === key,
        label: cards.length ? `${cards.map(c => rankName(c.rank)).join(' ')} (${classify(cards, game.level).name})` : 'pass',
      };
    });
    assert.ok(actions.some(action => action.expected));
    probes.push({ name, description: 'Reference card-choice diagnostic; optional asking excluded. Not proof of optimal strategy.',
      observation: observeFoundation(game), history: encodeFoundationHistory(game), actions });
  }
  add('Seven-card control then five-card exit', lateGame([53, 107, 5, 18, 31, 44, 7]), [53, 107]);
  add('Eight-card control then five-card exit', lateGame([53, 107, 161, 5, 18, 31, 44, 7]), [53, 107, 161]);
  add('Preserve mixed Jokers before the natural five-card exit', lateGame([53, 52, 5, 18, 31, 2, 15]), [52, 53]);
  const feed = lateGame([2, 15, 5, 18, 12, 25]);
  leadFor(feed, 2, [55, 68]);
  passTurn(feed, 1);
  add('Receive 33 with 44 and preserve 77 plus AA', feed, [2, 15]);
  const recapture = lateGame([2, 15, 12, 25, 53, 107]);
  leadFor(recapture, 2, [55, 68]);
  passTurn(recapture, 1);
  add('Receive a weak pair while retaining a strict recapture', recapture, [2, 15]);
  const stranded = lateGame([1, 14, 5, 18, 12, 25]);
  leadFor(stranded, 2, [56, 69]);
  passTurn(stranded, 1);
  add('Do not receive a pair feed that strands a lower pair', stranded, []);
  const waste = lateGame([53, 52, 5, 18, 31, 2, 15, 28]);
  leadFor(waste, 1, [60, 61, 62, 63, 64]);
  add('Do not burn both Jokers on a nonurgent low five-card exchange', waste, []);
  return probes;
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  process.stdout.write(`${JSON.stringify(tacticalProbes())}\n`);
}
