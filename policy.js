import { classify, beats, publicCounts } from './engine.js';

export const ENCODING_VERSION = 1;
export const OBSERVATION_SIZE = 922;
export const ACTION_SIZE = 71;
const ranks = Array.from({ length: 13 }, (_, i) => i + 2);
const windows = [[14, 2, 3, 4, 5], ...Array.from({ length: 9 }, (_, i) => Array.from({ length: 5 }, (_, j) => i + j + 2))];
const sizes = [0, 1, 2, 3, 5];
const cardType = c => c.rank > 14 ? c.rank === 19 ? 52 : 53 : c.suit * 13 + c.rank - 2;
const oneHot = (size, index) => Array.from({ length: size }, (_, i) => Number(i === index));

function cardFeatures(cards) {
  const features = Array(54).fill(0);
  for (const card of cards) features[cardType(card)] += 1 / 3;
  return features;
}

function typeFeatures(cards, level) {
  const type = cards.length ? classify(cards, level) : null;
  if (cards.length && !type) throw new Error('Invalid policy action');
  return [
    ...cardFeatures(cards), ...oneHot(5, sizes.indexOf(cards.length)),
    ...oneHot(7, type?.category ?? -1),
    ...Array.from({ length: 5 }, (_, i) => (type?.values[i] ?? 0) / 21),
  ];
}

export function observeForPolicy(game, player = game.turn) {
  const relative = seat => (seat - player + 6) % 6;
  const seats = Array.from({ length: 6 }, (_, i) => (player + i) % 6);
  const counts = publicCounts(game);
  if (game.phase === 'returning') for (const exchange of game.exchanges) {
    if (exchange.returned === null) { counts[exchange.from]--; counts[exchange.to]++; }
  }
  const features = cardFeatures(game.hands[player]);
  for (const seat of seats) features.push(...cardFeatures(game.played.filter(c => c.player === seat)));
  features.push(...seats.map(seat => counts[seat] / 27));
  features.push(...seats.map(seat => Number(game.passed.includes(seat))));
  features.push(...seats.map(seat => (game.order.indexOf(seat) + 1) / 6));
  features.push(...oneHot(13, game.level - 2), Number(game.dealer === player % 2));
  features.push(...typeFeatures(game.target?.cards ?? [], game.level));
  features.push(...oneHot(6, game.lastPlayer === null ? -1 : relative(game.lastPlayer)));
  features.push(Number(game.phase === 'playing'), Number(game.phase === 'returning'));
  const exchange = game.phase === 'returning' ? game.exchanges[game.returnIndex] : null;
  features.push(...cardFeatures(exchange?.to === player ? [exchange.tribute] : []));
  features.push(Number((game.direction ?? -1) === -1), ...oneHot(6, relative(game.lead)));
  const events = (game.policyEvents ?? game.history).slice(-6);
  for (let i = 0; i < 6; i++) {
    const event = events[i - (6 - events.length)];
    features.push(...cardFeatures(event?.cards ?? []));
    features.push(...oneHot(6, event ? relative(event.player) : -1));
    features.push(Number(event?.lead ?? false), Number(event?.pass ?? false));
  }
  if (features.length !== OBSERVATION_SIZE) throw new Error('Policy observation schema mismatch');
  return features;
}

export function recordPolicyEvent(game, player, previousHistoryLength, previousPhase) {
  if (previousPhase !== 'playing') return;
  if (!game.policyEvents) game.policyEvents = game.history.slice(0, previousHistoryLength).slice(-6);
  game.policyEvents.push(game.history.length > previousHistoryLength
    ? { ...game.history.at(-1), pass: false }
    : { player, cards: [], lead: false, pass: true });
  game.policyEvents = game.policyEvents.slice(-6);
}

export function legalPolicyActions(game, player = game.turn) {
  if (game.over) return [];
  if (player !== game.turn) throw new Error('Policy actions requested out of turn');
  const hand = game.hands[player];
  const byType = new Map();
  for (const card of [...hand].sort((a, b) => a.id - b.id)) {
    const key = cardType(card);
    if (!byType.has(key)) byType.set(key, []);
    byType.get(key).push(card);
  }
  const types = [...byType.keys()].sort((a, b) => a - b);
  const rankTypes = new Map(ranks.map(rank => [rank, types.filter(t => byType.get(t)[0].rank === rank)]));
  const wildTypes = types.filter(t => t >= 52);
  const available = (type, used) => byType.get(type).length - (used.get(type) ?? 0);
  const found = new Map();
  function materialize(used) {
    return [...used].flatMap(([type, amount]) => byType.get(type).slice(0, amount));
  }
  function add(used) {
    const cards = materialize(used);
    const type = classify(cards, game.level);
    if (!beats(type, game.target?.type)) return;
    const ids = cards.map(c => c.id).sort((a, b) => a - b);
    found.set(ids.join(','), { kind: 'play', ids, features: typeFeatures(cards, game.level) });
  }
  function choose(pool, count, used, callback, index = 0) {
    if (count === 0) return callback(used);
    if (index === pool.length) return;
    const type = pool[index];
    const old = used.get(type) ?? 0;
    const max = Math.min(count, available(type, used));
    for (let n = 0; n <= max; n++) {
      if (old + n) used.set(type, old + n);
      else used.delete(type);
      choose(pool, count - n, used, callback, index + 1);
    }
    if (old) used.set(type, old);
    else used.delete(type);
  }
  function group(rank, count, used, callback) {
    const wildCount = wildTypes.reduce((n, t) => n + available(t, used), 0);
    for (let n = 0; n <= Math.min(count, wildCount); n++) {
      choose(rankTypes.get(rank), count - n, used, selected => choose(wildTypes, n, selected, callback));
    }
  }
  if (game.phase === 'returning') {
    const forbidden = game.exchanges[game.returnIndex].tribute.id;
    return types.flatMap(type => {
      const card = byType.get(type).find(c => c.id !== forbidden);
      return card ? [{ kind: 'return', ids: [card.id], features: typeFeatures([card], game.level) }] : [];
    });
  }
  const requested = game.target ? [game.target.type.count] : [1, 2, 3, 5];
  if (requested.includes(1)) for (const type of types) add(new Map([[type, 1]]));
  for (const size of requested.filter(n => n > 1)) {
    for (const rank of ranks) group(rank, size, new Map(), add);
    choose(wildTypes, size, new Map(), add);
  }
  if (requested.includes(5)) {
    for (const rank of ranks) {
      group(rank, 4, new Map(), used => {
        for (const type of types) {
          if (available(type, used) <= 0) continue;
          const old = used.get(type) ?? 0;
          used.set(type, old + 1);
          add(used);
          if (old) used.set(type, old);
          else used.delete(type);
        }
      });
      group(rank, 3, new Map(), used => {
        for (const other of ranks) if (rank !== other) group(other, 2, used, add);
      });
    }
    for (const window of windows) {
      function run(index, used) {
        if (index === 5) return add(used);
        for (const type of [...rankTypes.get(window[index]), ...wildTypes]) {
          if (available(type, used) <= 0) continue;
          const old = used.get(type) ?? 0;
          used.set(type, old + 1);
          run(index + 1, used);
          if (old) used.set(type, old);
          else used.delete(type);
        }
      }
      run(0, new Map());
    }
    for (let suit = 0; suit < 4; suit++) {
      const suited = types.filter(t => t < 52 && Math.floor(t / 13) === suit);
      const wildCount = wildTypes.reduce((n, type) => n + byType.get(type).length, 0);
      for (let n = 0; n <= Math.min(5, wildCount); n++) {
        choose(suited, 5 - n, new Map(), used => choose(wildTypes, n, used, add));
      }
    }
  }
  const actions = [...found.values()];
  if (game.target) actions.push({ kind: 'pass', ids: [], features: typeFeatures([], game.level) });
  if (!actions.length) throw new Error('No legal policy actions');
  return actions;
}
