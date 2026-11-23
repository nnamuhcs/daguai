import test from 'node:test';
import assert from 'node:assert/strict';
import { createDeck, newGame, classify, beats, candidates, playCards, passTurn, computerTurn, validSave, canAsk, askCards, returnCard, planHand, suggestion, secureRandomIndex } from './engine.js';
import { lateGame, leadFor } from './training/foundation-scenarios.js';

let id = 1000;
const c = (rank, suit = 0) => ({ id: id++, rank, suit: rank > 14 ? -1 : suit });

test('three complete decks, six equal unique hands', () => {
  const game = newGame();
  assert.equal(createDeck().length, 162);
  assert.ok(game.hands.every(h => h.length === 27));
  assert.equal(new Set(game.hands.flat().map(c => c.id)).size, 162);
  assert.equal(validSave(game), true);
  game.hands[0].push(game.hands[1][0]);
  assert.equal(validSave(game), false);
});

test('secure random draws discard the biased tail and seeded shuffles remain reproducible', t => {
  const values = [2 ** 32 - 1, 7];
  let calls = 0;
  t.mock.method(globalThis.crypto, 'getRandomValues', buffer => {
    buffer[0] = values[calls++];
    return buffer;
  });
  assert.equal(secureRandomIndex(6), 1);
  assert.equal(calls, 2);
  assert.throws(() => secureRandomIndex(0), /随机范围/);
  const deterministic = () => {
    let seed = 42;
    return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  };
  assert.deepEqual(newGame(null, deterministic()).hands, newGame(null, deterministic()).hands);
});

test('wildcards, trump strength, and minimum straight interpretation', () => {
  assert.equal(classify([c(8), c(19)], 8).values[0], 18);
  assert.equal(classify([c(19), c(20)]).values[0], 20);
  assert.equal(classify([c(3), c(4), c(5), c(6), c(20)]).name, '同花顺');
  assert.equal(classify([c(3), c(4), c(5), c(6), c(20)]).values[0], 6);
  assert.equal(classify([c(14), c(2), c(3), c(4), c(5)]).values[0], 5);
  assert.equal(classify([c(12), c(13, 1), c(14), c(2), c(3)]), null);
  assert.equal(classify([c(8), c(8), c(8), c(20), c(4)]).name, '四带一');
  assert.equal(classify([c(7), c(7), c(7), c(19), c(20)]).name, '五同');
  assert.equal(classify([c(2), c(3)]), null);
  assert.equal(classify([c(2), c(2), c(2), c(2)]), null);
});

test('five card hierarchy never beats a single, pair, or triple', () => {
  const five = classify([c(3), c(3), c(3), c(3), c(3)]);
  const straight = classify([c(10), c(11, 1), c(12), c(13), c(14)]);
  assert.equal(beats(five, straight), true);
  assert.equal(beats(five, classify([c(2)])), false);
  assert.equal(beats(classify([c(9), c(9)]), classify([c(9), c(9)])), false);
});

test('reject illegal plays without mutating state; candidates use owned unique cards', () => {
  const game = newGame();
  const before = JSON.stringify(game);
  assert.throws(() => passTurn(game, game.turn), /领出/);
  assert.throws(() => playCards(game, game.turn, [-1]), /不在手牌/);
  assert.equal(JSON.stringify(game), before);
  const hand = [c(3), c(4), c(5), c(6), c(19), c(19), c(7, 2), c(7, 2), c(7)];
  for (const move of candidates(hand, 2)) {
    assert.equal(new Set(move.cards.map(c => c.id)).size, move.cards.length);
    assert.ok(move.cards.every(c => hand.includes(c)));
    assert.ok(classify(move.cards));
  }
});

test('finished leader gives the next lead to the next active player, even an opponent', () => {
  const game = newGame();
  game.hands = Array.from({ length: 6 }, (_, i) => [c(i + 3)]);
  playCards(game, 0, [game.hands[0][0].id]);
  for (const player of [5, 4, 3, 2, 1]) passTurn(game, player);
  assert.equal(game.turn, 5);
  assert.equal(game.target, null);
  assert.deepEqual(game.order, [0]);
});

test('many complete automated games terminate and preserve cards', () => {
  let seed = 17;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let round = 0; round < 20; round++) {
    const game = newGame(null, random);
    let turns = 0;
    while (!game.over && turns++ < 1500) {
      computerTurn(game);
      assert.ok(validSave(game));
    }
    assert.ok(game.over, `game ${round} did not terminate`);
    assert.equal(new Set(game.order).size, 6);
    assert.ok(game.result);
  }
});

test('dealer upgrades on a win; challenger takes over without upgrading', () => {
  for (const dealer of [0, 1]) {
    const game = newGame();
    game.dealer = dealer;
    game.order = [0, 2];
    game.hands = [[], [c(5)], [], [c(6)], [c(3)], [c(4), c(4)]];
    game.turn = 4;
    playCards(game, 4, [game.hands[4][0].id]);
    assert.equal(game.result.winner, 0);
    assert.equal(game.result.sweep, true);
    assert.equal(game.dealer, 0);
    assert.equal(game.levels[0], dealer === 0 ? 3 : 2);
    const next = newGame(game);
    assert.equal(next.level, dealer === 0 ? 3 : 2);
  }
});

test('asking is turn-gated, requires beating the previous play, and never reveals composition', () => {
  const game = newGame();
  game.hands[0] = [c(3)];
  game.hands[1] = Array.from({ length: 11 }, () => c(4));
  game.target = { cards: [c(5)], type: classify([c(5)]) };
  game.lastPlayer = 1;
  assert.equal(canAsk(game), false);
  assert.throws(() => askCards(game), /能压过/);
  game.hands[0] = [c(6)];
  assert.equal(canAsk(game), true);
  assert.deepEqual(askCards(game), { player: 1, count: null });
  game.hands[1].pop();
  assert.deepEqual(askCards(game), { player: 1, count: 10 });
  assert.equal(canAsk(game, 2), false);
  assert.throws(() => askCards(game, 0, 3), /上一出牌者/);
  game.target = null;
  assert.equal(canAsk(game, 0, 1), false);
  assert.equal(canAsk(game, 0, 2), true);
  assert.equal(askCards(game, 0, 2).player, 2);
});

test('a team finishing stops the round immediately and traps every remaining opponent', () => {
  const game = newGame();
  game.order = [0, 2];
  game.turn = 4;
  game.hands[0] = [];
  game.hands[2] = [];
  game.hands[4] = [c(9)];
  playCards(game, 4, [game.hands[4][0].id]);
  assert.equal(game.over, true);
  assert.equal(game.result.finishedCount, 3);
  assert.deepEqual(game.result.trapped, [5, 3, 1]);
  assert.equal(game.result.sweep, true);
  assert.ok(game.hands[1].length > 0);
});

test('single, double and triple tribute are maximum cards, paired once, and balanced after returns', () => {
  for (const trapped of [[5], [3, 5], [1, 3, 5]]) {
    const previous = newGame();
    previous.over = true;
    previous.order = [0, 2, 4, ...trapped];
    previous.result = { winner: 0, trapped };
    const game = newGame(previous);
    assert.equal(game.phase, 'returning');
    assert.equal(game.exchanges.length, trapped.length);
    assert.equal(new Set(game.exchanges.map(e => e.to)).size, trapped.length);
    assert.equal(validSave(game), true);
    for (const exchange of game.exchanges) {
      const donorMaximum = Math.max(...game.hands[exchange.from].map(c => c.rank > 14 ? c.rank + 1 : c.rank === game.level ? 18 : c.rank));
      const donatedStrength = exchange.tribute.rank > 14 ? exchange.tribute.rank + 1 : exchange.tribute.rank === game.level ? 18 : exchange.tribute.rank;
      assert.ok(donatedStrength >= donorMaximum);
      assert.throws(() => returnCard(game, exchange.to, exchange.tribute.id), /原样退回/);
      assert.throws(() => playCards(game, game.turn, [game.hands[game.turn][0].id]), /轮到/);
      const card = game.hands[exchange.to].find(c => c.id !== exchange.tribute.id);
      returnCard(game, exchange.to, card.id);
      assert.equal(validSave(game), true);
    }
    assert.equal(game.phase, 'playing');
    assert.equal(game.turn, 0);
    assert.ok(game.hands.every(h => h.length === 27));
  }
});

test('planner preserves a complete multi-road exit and strategy does not inspect opponents', () => {
  const hand = [c(3), c(3), c(3), c(4), c(4), c(5), c(5), c(5), c(6), c(6), c(20)];
  const plan = planHand(hand);
  assert.equal(plan.turns, 3);
  assert.equal(new Set(plan.moves.flat().map(c => c.id)).size, hand.length);
  assert.ok(plan.moves.every(move => classify(move)));
  const game = newGame();
  const first = suggestion(game)?.map(c => c.id);
  for (let i = 1; i < 6; i++) game.hands[i] = [c(20), c(20), c(20)];
  assert.deepEqual(suggestion(game)?.map(c => c.id), first);
});

test('new rounds after completed computer games perform tribute and resume correctly', () => {
  let game = newGame();
  for (let round = 0; round < 4; round++) {
    let turns = 0;
    while (!game.over && turns++ < 1200) {
      computerTurn(game);
      assert.equal(validSave(game), true);
    }
    assert.ok(game.over);
    game = newGame(game);
    assert.ok(validSave(game));
  }
});

test('counterclockwise play and pass cycle wraps around and returns lead to the unbeaten player', () => {
  const game = newGame();
  playCards(game, 0, [game.hands[0][0].id]);
  for (const player of [5, 4, 3, 2, 1]) {
    assert.equal(game.turn, player);
    passTurn(game, player);
  }
  assert.equal(game.turn, 0);
  assert.equal(game.target, null);
  game.hands[5] = [];
  playCards(game, 0, [game.hands[0][0].id]);
  assert.equal(game.turn, 4);
});

test('legacy in-progress trick finishes its existing cycle before changing direction', () => {
  const game = newGame();
  game.direction = 1;
  game.nextDirection = -1;
  game.hands = Array.from({ length: 6 }, (_, i) => [c(i + 3)]);
  playCards(game, 0, [game.hands[0][0].id]);
  for (const player of [1, 2, 3, 4, 5]) {
    assert.equal(game.turn, player);
    passTurn(game, player);
  }
  assert.equal(game.direction, -1);
  assert.equal(game.nextDirection, undefined);
  assert.equal(game.turn, 5);
});

test('multiple tribute pairings follow counterclockwise seating distance', () => {
  const previous = newGame();
  previous.order = [0, 2, 4, 5, 3, 1];
  previous.result = { winner: 0, trapped: [5, 3, 1] };
  const game = newGame(previous);
  assert.deepEqual(game.exchanges.map(e => [e.from, e.to]), [[5, 4], [3, 2], [1, 0]]);
});

test('AA plus a joker is three aces, which lose to three current trumps', () => {
  const aces = classify([c(14), c(14), c(20)], 3);
  const trumps = classify([c(3), c(3), c(3)], 3);
  assert.equal(aces.name, '三张');
  assert.equal(aces.values[0], 14);
  assert.equal(trumps.values[0], 18);
  assert.equal(beats(trumps, aces), true);
});

test('AI does not overtake an ally just because an enemy has a different one-hand card count', () => {
  const game = newGame();
  game.hands[0] = [c(2), c(2), c(2), c(4)];
  game.target = { cards: [c(14), c(14), c(14)], type: classify([c(14), c(14), c(14)]) };
  game.lastPlayer = 2;
  game.played = Array.from({ length: 26 }, () => ({ ...c(5), player: 1 }));
  assert.equal(suggestion(game, 0), null);
});

test('AI preserves joker combinations rather than casually raising a teammate', () => {
  const game = newGame();
  game.hands[0] = [c(20), c(20), c(6), c(6), c(4)];
  game.target = { cards: [c(14), c(14)], type: classify([c(14), c(14)]) };
  game.lastPlayer = 2;
  game.played = Array.from({ length: 25 }, () => ({ ...c(5), player: 1 }));
  assert.equal(suggestion(game, 0), null);
});

test('AI may finish its whole hand over an ally', () => {
  const game = newGame();
  game.hands[0] = [c(20)];
  game.target = { cards: [c(14)], type: classify([c(14)]) };
  game.lastPlayer = 2;
  assert.deepEqual(suggestion(game, 0), game.hands[0]);
  computerTurn(game);
  assert.equal(game.history.at(-1).reason, '一次出完，不留尾牌');
});

test('AI can defend against immediately handing a finished ally lead to a one-card opponent', () => {
  const game = newGame();
  const deck = createDeck();
  const own = [deck[53], deck[2]];
  const enemy = deck[52];
  const remaining = deck.filter(c => ![53, 2, 52].includes(c.id));
  game.hands = [own, [enemy], [], [], [], []];
  game.played = [];
  let offset = 0;
  for (const player of [0, 1, 2, 3, 4, 5]) {
    const count = 27 - game.hands[player].length;
    game.played.push(...remaining.slice(offset, offset + count).map(c => ({ ...c, player })));
    offset += count;
  }
  const ace = game.played.find(c => c.player === 2 && c.rank === 14);
  game.target = { cards: [ace], type: classify([ace]) };
  game.lastPlayer = 2;
  game.order = [3, 4, 5, 2];
  game.passed = [1];
  assert.deepEqual(suggestion(game, 0), [deck[53]]);
});

test('seven cards: unbeatable joker pair leads before the weak five-card hand', () => {
  const game = lateGame([53, 107, 5, 18, 31, 44, 7]);
  const move = suggestion(game);
  assert.equal(move.length, 2);
  assert.ok(move.every(c => c.rank === 20));
  assert.ok(classify(game.hands[0].filter(c => !move.includes(c))));
});

test('eight cards: unbeatable joker triple leads before the weak five-card hand', () => {
  const game = lateGame([53, 107, 161, 5, 18, 31, 44, 7]);
  const move = suggestion(game);
  assert.equal(move.length, 3);
  assert.ok(move.every(c => c.rank === 20));
  assert.ok(classify(game.hands[0].filter(c => !move.includes(c))));
});

test('seven and eight cards do not imply a legal two-hand exit', () => {
  const seven = lateGame([5, 18, 31, 2, 15, 28, 7]);
  const eight = lateGame([1, 14, 4, 17, 7, 20, 35, 50]);
  assert.equal(planHand(seven.hands[0]).turns, 3);
  assert.equal(planHand(eight.hands[0]).turns, 5);
});

test('AI does not spend both jokers to lead a weak five-of-a-kind', () => {
  const game = lateGame([53, 52, 5, 18, 31, 2, 15, 28]);
  const move = suggestion(game);
  assert.ok(!(move.length === 5 && move.filter(c => c.rank > 14).length >= 2));
});

test('AI may pass rather than spend both jokers on a nonurgent five-card exchange', () => {
  const game = lateGame([53, 52, 5, 18, 31, 2, 15, 28]);
  leadFor(game, 1, [60, 61, 62, 63, 64]);
  assert.equal(suggestion(game), null);
  const removed = game.hands[1].splice(5);
  game.played.push(...removed.map(c => ({ ...c, player: 1 })));
  assert.ok(validSave(game));
  const defensive = suggestion(game);
  assert.ok(defensive && beats(classify(defensive), game.target.type));
});

test('AI accepts teammate pair or triple feeds when it can control and finish the actual remainder', () => {
  for (const size of [2, 3]) {
    const game = lateGame([53, 107, ...(size === 3 ? [161] : []), 5, 18, 31, 44, 7]);
    leadFor(game, 2, [55, 68, 81].slice(0, size));
    passTurn(game, 1);
    const move = suggestion(game);
    assert.equal(move?.length, size);
    assert.ok(move.every(c => c.rank === 20));
    computerTurn(game);
    assert.equal(game.history.at(-1).reason, '接同伴送牌，两手收完');
    while (game.target) passTurn(game, game.turn);
    assert.equal(game.turn, 0);
    computerTurn(game);
    assert.equal(game.hands[0].length, 0);
    assert.ok(validSave(game));
  }
});

test('AI accepts a weak teammate feed with a same-road recapture and legal final hand', () => {
  const game = lateGame([2, 15, 12, 25, 53, 107]);
  leadFor(game, 2, [55, 68]);
  passTurn(game, 1);
  const move = suggestion(game);
  assert.deepEqual(move?.map(c => c.rank), [4, 4]);
  computerTurn(game);
  assert.equal(game.history.at(-1).reason, '接同伴送牌，留同路回手再收尾');
  leadFor(game, 5, [58, 71]);
  passTurn(game, 4);
  passTurn(game, 3);
  passTurn(game, 2);
  passTurn(game, 1);
  computerTurn(game);
  assert.ok(game.target.cards.every(c => c.rank === 20));
  while (game.target) passTurn(game, game.turn);
  computerTurn(game);
  assert.equal(game.hands[0].length, 0);
  assert.ok(validSave(game));
});

test('AI does not claim a teammate-assisted recapture when unseen opponents can match or beat its control', () => {
  const game = lateGame([2, 15, 12, 25, 53, 52]);
  leadFor(game, 2, [55, 68]);
  passTurn(game, 1);
  assert.equal(suggestion(game), null);
});

test('AI accepts a cheap teammate feed along its actual pair route without spending the higher pairs', () => {
  const game = lateGame([2, 15, 5, 18, 12, 25]);
  leadFor(game, 2, [55, 68]);
  passTurn(game, 1);
  assert.deepEqual(suggestion(game)?.map(c => c.rank), [4, 4]);
  computerTurn(game);
  assert.equal(game.history.at(-1).reason, '接同伴送牌，继续对子路');
  const rest = planHand(game.hands[0]);
  assert.ok(rest.moves.every(cards => cards.length === 2));
  assert.ok(validSave(game));
});

test('a teammate feed does not justify leaving a lower pair stranded or claiming equal jokers can recapture', () => {
  const stranded = lateGame([1, 14, 5, 18, 12, 25]);
  leadFor(stranded, 2, [56, 69]);
  passTurn(stranded, 1);
  assert.equal(suggestion(stranded), null);
  const equal = lateGame([2, 12, 53]);
  leadFor(equal, 2, [55]);
  passTurn(equal, 1);
  assert.equal(suggestion(equal), null);
});

test('mixed joker control leads before an actual five-card remainder instead of wasting both on five sevens', () => {
  const game = lateGame([53, 52, 5, 18, 31, 2, 15]);
  const move = suggestion(game);
  assert.equal(move.length, 2);
  assert.ok(move.every(c => c.rank > 14));
  assert.ok(classify(game.hands[0].filter(c => !move.includes(c))));
});

test('local policy overrides are validated by the engine before any state mutation', () => {
  const game = newGame();
  const before = JSON.stringify(game);
  assert.throws(() => computerTurn(game, [999]), /手牌/);
  assert.throws(() => computerTurn(game, []), /不合法/);
  assert.equal(JSON.stringify(game), before);
  const move = candidates(game.hands[0], game.level)[0];
  computerTurn(game, move.cards.map(c => c.id));
  assert.equal(game.history.at(-1).reason, '本地模型选择（试验）');
  assert.ok(validSave(game));
});
