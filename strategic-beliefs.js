import { beats, candidates, classify, createDeck, publicCounts, sortHand, strength } from './engine.js';

const deck = createDeck();
const type = card => card.rank > 14 ? card.rank + 33 : card.suit * 13 + card.rank - 2;
const seats = [0, 1, 2, 3, 4, 5];

export function beliefInformation(game) {
  if (game.over) throw new Error('Beliefs require a live decision');
  const actor = game.turn, counts = publicCounts(game);
  const exchanges = game.exchanges.map(exchange => {
    if (!exchange.returned) { counts[exchange.from]--; counts[exchange.to]++; }
    const involved = actor === exchange.from || actor === exchange.to;
    return {
      from: exchange.from, to: exchange.to, returned: Boolean(exchange.returned),
      tributeRank: exchange.tribute.rank,
      tributeType: involved ? type(exchange.tribute) : null,
      returnType: involved && exchange.returned ? type(exchange.returned) : null,
    };
  });
  return {
    actor, counts, level: game.level, own: game.hands[actor].map(type).sort((a, b) => a - b),
    played: game.played.map(card => [card.player, type(card)]),
    questions: game.questions.filter(q => q.target).map(q => ({
      player: q.player, playedAt: q.playedAt,
      target: { count: q.target.count, category: q.target.category, values: [...q.target.values] },
    })),
    exchanges,
  };
}

function prepare(info) {
  const seen = Array(54).fill(0);
  const take = key => {
    if (!Number.isInteger(key) || key < 0 || key >= 54 || seen[key] >= 3) {
      throw new Error('Belief information violates deck conservation');
    }
    return deck[key + 54 * seen[key]++];
  };
  const own = info.own.map(take);
  const played = info.played.map(([seat, key]) => ({ ...take(key), player: seat }));
  const past = seats.map(seat => played.filter(card => card.player === seat));
  const unknown = deck.filter(card => Math.floor(card.id / 54) >= seen[type(card)]);
  const capacity = info.counts.reduce((sum, value) => sum + value, 0) - own.length;
  if (info.counts.length !== 6 || info.counts[info.actor] !== own.length || capacity !== unknown.length ||
      info.counts.some(n => !Number.isInteger(n) || n < 0 || n > 28)) {
    throw new Error('Belief information violates public hand capacities');
  }
  const checks = info.questions.map(q => ({
    ...q, since: played.slice(q.playedAt).filter(card => card.player === q.player),
  }));
  return { own, past, unknown, checks };
}

function transfersPossible(hands, past, info) {
  for (const exchange of info.exchanges) {
    const donor = [...hands[exchange.from], ...past[exchange.from]];
    const receiver = [...hands[exchange.to], ...past[exchange.to]];
    // When duplicate types exist, preserve uncertainty about which copy was exchanged.
    const returns = exchange.returned ? donor.filter(c =>
      exchange.returnType === null || type(c) === exchange.returnType) : [null];
    const tributes = receiver.filter(c => c.rank === exchange.tributeRank &&
      (exchange.tributeType === null || type(c) === exchange.tributeType));
    const higherThan = (card, tribute) => strength(card.rank, info.level) > strength(tribute.rank, info.level) ||
      (card.rank === tribute.rank && card.suit < tribute.suit);
    if (!tributes.some(tribute => returns.some(returned =>
      donor.every(card => card === returned || !higherThan(card, tribute))))) return false;
  }
  return true;
}

export function sampleBeliefWorlds(info, { samples = 16, attempts = 128, seed = 1,
  useEvidence = true } = {}) {
  if (!Number.isInteger(samples) || samples < 1 || samples > 64 || !Number.isInteger(attempts) ||
      attempts < samples || attempts > 4096 || !Number.isSafeInteger(seed)) {
    throw new Error('Invalid bounded belief sampling budget');
  }
  const { own, past, unknown, checks } = prepare(info);
  const worlds = [];
  let tried = 0, rng = seed >>> 0;
  const random = () => ((rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0) / 2 ** 32);
  while (tried < attempts && worlds.length < samples) {
    tried++;
    const cards = [...unknown];
    for (let i = cards.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [cards[i], cards[j]] = [cards[j], cards[i]];
    }
    let offset = 0;
    const hands = seats.map(seat => {
      if (seat === info.actor) return own;
      const hand = cards.slice(offset, offset + info.counts[seat]);
      offset += hand.length;
      return sortHand(hand, info.level);
    });
    if (useEvidence && (!transfersPossible(hands, past, info) ||
      checks.some(q => !candidates(sortHand([...hands[q.player], ...q.since], info.level),
        info.level, q.target).length))) continue;
    worlds.push(hands);
  }
  return { worlds, tried };
}

export function summarizeBeliefs(info, sampled, target = null) {
  const { worlds, tried } = sampled;
  const size = worlds.length;
  const cards = Array.from({ length: 6 }, () => Array(54).fill(0));
  const finish = seats.map(() => 0), beat = seats.map(() => 0);
  for (const world of worlds) {
    for (const seat of seats) {
      const hand = world[seat];
      for (const card of hand) cards[seat][type(card)]++;
      if (classify(hand, info.level)) finish[seat]++;
      if (target && candidates(hand, info.level, target).some(move => beats(move.type, target))) beat[seat]++;
    }
  }
  return {
    samples: size, attempts: tried,
    expectedCounts: size ? cards.map(row => row.map(n => n / size)) : null,
    finishInOneIfLeading: size ? finish.map(n => n / size) : null,
    canOvertake: size && target ? beat.map(n => n / size) : null,
  };
}
