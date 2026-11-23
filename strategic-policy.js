import { classify, publicCounts, strength } from './engine.js';
import manifest from './training/manifests/strategic-v3.json' with { type: 'json' };

export { manifest };
const typeIndex = card => card.rank > 14 ? card.rank + 33 : card.suit * 13 + card.rank - 2;
const windows = [[14, 2, 3, 4, 5],
  ...Array.from({ length: 9 }, (_, start) => Array.from({ length: 5 }, (_, i) => start + i + 2))];

function exactPartitions(hand, level) {
  const options = hand.map(() => []);
  const memo = new Map([[0, 0]]);
  function generate(start, cards, mask) {
    if (cards.length === 5 || [1, 2, 3].includes(cards.length)) {
      if (classify(cards, level)) {
        const first = 31 - Math.clz32(mask & -mask);
        options[first].push([mask, cards.length]);
        memo.set(mask, 1);
      }
      if (cards.length === 5) return;
    }
    for (let i = start; i < hand.length; i++) {
      cards.push(hand[i]);
      generate(i + 1, cards, mask | 1 << i);
      cards.pop();
    }
  }
  generate(0, [], 0);
  for (const group of options) group.sort((a, b) => b[1] - a[1]);
  function solve(mask) {
    if (memo.has(mask)) return memo.get(mask);
    const first = 31 - Math.clz32(mask & -mask);
    let best = hand.length;
    let count = 0;
    for (let rest = mask; rest; rest &= rest - 1) count++;
    for (const [move, size] of options[first]) {
      if ((move & mask) !== move || 1 + Math.ceil((count - size) / 5) >= best) continue;
      best = Math.min(best, 1 + solve(mask ^ move));
    }
    memo.set(mask, best);
    return best;
  }
  return solve;
}

function describe(hand, level, turns = null) {
  const ranks = Array(15).fill(0), suits = Array(4).fill(0);
  let small = 0, big = 0, highest = 0;
  for (const card of hand) {
    highest = Math.max(highest, strength(card.rank, level));
    if (card.rank === 19) small++;
    else if (card.rank === 20) big++;
    else { ranks[card.rank]++; suits[card.suit]++; }
  }
  const wild = small + big, highGroup = Math.max(...ranks);
  // Five-card categories are rank groups, same-suit cards, or rank runs.
  const five = hand.length >= 5 && (highGroup + wild >= 4 ||
    Math.max(...suits) + wild >= 5 ||
    windows.some(run => run.filter(rank => !ranks[rank]).length <= wild) ||
    ranks.some((n, first) => first >= 2 && ranks.some((m, second) =>
      second >= 2 && second !== first && Math.max(0, 3 - n) + Math.max(0, 2 - m) <= wild)));
  const cover = Boolean(classify(hand, level));
  // A constructive fallback: partition natural ranks separately; play Jokers
  // together, except four (triple + single) or six (five + single).
  const groupTurns = n => Math.floor(n / 5) + (n % 5 === 4 ? 2 : Number(n % 5 > 0));
  return { count: hand.length, small, big, trump: ranks[level], highest, cover, five,
    pair: highGroup + wild >= 2, triple: highGroup + wild >= 3,
    turns: turns ?? (cover ? 1 : ranks.reduce((n, size) => n + groupTurns(size), groupTurns(wild))) };
}

function transferMemory(game, player) {
  const participating = game.exchanges.filter(exchange => exchange.from === player || exchange.to === player);
  if (participating.length > 1) throw new Error('Strategic schema permits one tribute/return pair per seat');
  const exchange = participating[0];
  const event = (card, from, to) => {
    const row = Array(62).fill(0);
    if (!card) return row;
    const sent = from === player;
    row[typeIndex(card)] = 1;
    row[54 + ((sent ? to : from) - player + 6) % 6] = 1;
    row[sent ? 60 : 61] = 1;
    return row;
  };
  return [event(exchange?.tribute, exchange?.from, exchange?.to),
    event(exchange?.returned, exchange?.to, exchange?.from)].flat();
}

export function strategicFeatures(game, legal, player = game.turn) {
  if (game.over || player !== game.turn) throw new Error('Strategic features require the live acting seat');
  const hand = game.hands[player], { level, target, lastPlayer } = game;
  const index = new Map(hand.map((card, i) => [card.id, i]));
  const partition = hand.length <= manifest.smallHandExactLimit ? exactPartitions(hand, level) : null;
  const fullMask = (1 << hand.length) - 1;
  const before = describe(hand, level, partition?.(fullMask));
  const counts = publicCounts(game);
  for (const exchange of game.exchanges) {
    if (!exchange.returned) { counts[exchange.from]--; counts[exchange.to]++; }
  }
  const ally = seat => seat % 2 === player % 2;
  const liveMin = sameTeam => {
    const live = counts.filter((n, seat) => n > 0 && seat !== player && ally(seat) === sameTeam);
    return live.length ? Math.min(...live) : 0;
  };
  const minAlly = liveMin(true), minEnemy = liveMin(false);
  const next = Array.from({ length: 5 }, (_, i) => (player + (game.direction ?? -1) * (i + 1) + 6) % 6)
    .find(seat => counts[seat] > 0);
  const finishedTeammates = game.order.filter(ally).length;
  const actionExtras = legal.map(action => {
    let mask = 0;
    const cards = action.ids.map(id => {
      const i = index.get(id);
      if (i === undefined || mask & 1 << i) throw new Error('Strategic action must use distinct own cards');
      mask |= 1 << i;
      return hand[i];
    });
    const after = mask ? describe(hand.filter((_, i) => !(mask & 1 << i)), level,
      partition?.(fullMask ^ mask)) : before;
    const play = action.kind === 'play', ends = play && after.count === 0;
    return [
      before.count / 27, after.count / 27, cards.length / 5, (before.small - after.small) / 3,
      (before.big - after.big) / 3, (before.trump - after.trump) / 12,
      after.small / 3, after.big / 3, after.trump / 12,
      play ? action.features.slice(59, 66).indexOf(1) / 6 : 0,
      Number(play && target && ally(lastPlayer)), Number(play && target && !ally(lastPlayer)),
      Number(ends), Number(ends && finishedTeammates === 2),
      minAlly / 27, minEnemy / 27, Number(after.cover), Math.min(after.turns, 6) / 6,
      Number(after.pair), Number(after.triple), Number(after.five), after.highest / 21,
      Number(ends && next !== undefined && ally(next)), (target?.type.count ?? 0) / 5,
    ];
  });
  return { observation: [
    before.count / 27, Number(before.cover), partition ? Number(before.turns === 2) : -1,
    Math.min(before.turns, 6) / 6, Number(Boolean(partition)), Number(before.five),
    before.highest / 21, before.trump / 12, ...transferMemory(game, player),
  ], actionExtras };
}
