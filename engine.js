export const SUITS = ['♠', '♥', '♣', '♦'];
export const NAMES = ['你', '阿诚', '小满', '老周', '阿岚', '小林'];
export const rankName = rank => ({ 11: 'J', 12: 'Q', 13: 'K', 14: 'A', 19: '小怪', 20: '大怪' }[rank] || String(rank));
export const strength = (rank, level) => rank > 14 ? rank + 1 : rank === level ? 18 : rank;
const ranks = Array.from({ length: 13 }, (_, i) => i + 2);
const windows = [[14, 2, 3, 4, 5], ...Array.from({ length: 9 }, (_, i) => Array.from({ length: 5 }, (_, j) => i + j + 2))];
export const sortHand = (cards, level) => [...cards].sort((a, b) => strength(b.rank, level) - strength(a.rank, level) || a.suit - b.suit || a.id - b.id);

export function createDeck() {
  return Array.from({ length: 162 }, (_, id) => {
    const n = id % 54;
    return { id, rank: n < 52 ? n % 13 + 2 : n === 52 ? 19 : 20, suit: n < 52 ? Math.floor(n / 13) : -1 };
  });
}

const compare = (a, b) => {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const delta = (a[i] || 0) - (b[i] || 0);
    if (delta) return delta;
  }
  return 0;
};

export function classify(cards, level = 2) {
  const count = cards.length;
  if (![1, 2, 3, 5].includes(count)) return null;
  const natural = cards.filter(c => c.rank < 19);
  const wild = count - natural.length;
  const group = new Map();
  natural.forEach(c => group.set(c.rank, (group.get(c.rank) || 0) + 1));
  const result = (category, name, values) => ({ count, category, name, values });
  if (count === 1) return result(0, '单张', [strength(cards[0].rank, level)]);
  if (group.size <= 1) {
    const rank = natural[0]?.rank ?? (cards.every(c => c.rank === 20) ? 20 : 19);
    return result(count === 5 ? 6 : 0, count === 5 ? '五同' : count === 3 ? '三张' : '对子', [strength(rank, level)]);
  }
  if (count !== 5) return null;
  const straight = windows.find(w => natural.every(c => w.includes(c.rank)) && group.size === natural.length);
  const sameSuit = natural.every(c => c.suit === natural[0].suit);
  if (straight && sameSuit) return result(5, '同花顺', [straight[4]]);
  const quads = ranks.filter(r => (group.get(r) || 0) + wild >= 4 && natural.length - (group.get(r) || 0) <= 1);
  if (quads.length) return result(4, '四带一', [Math.min(...quads.map(r => strength(r, level)))]);
  const triples = ranks.filter(r => {
    if ((group.get(r) || 0) > 3) return false;
    return ranks.some(s => s !== r && (group.get(s) || 0) <= 2 &&
      [...group.keys()].every(k => k === r || k === s) &&
      3 - (group.get(r) || 0) + 2 - (group.get(s) || 0) === wild);
  });
  if (triples.length) return result(3, '三带二', [Math.min(...triples.map(r => strength(r, level)))]);
  if (sameSuit) {
    return result(2, '同花', [...natural.map(c => strength(c.rank, level)), ...Array(wild).fill(Math.min(...ranks.map(r => strength(r, level))))].sort((a, b) => b - a));
  }
  if (straight) return result(1, '杂顺', [straight[4]]);
  return null;
}

export function beats(play, target) {
  return !!play && (!target || play.count === target.count &&
    (play.category > target.category || play.category === target.category && compare(play.values, target.values) > 0));
}

export function candidates(hand, level, target = null) {
  const found = new Map();
  const wild = hand.filter(c => c.rank > 14).sort((a, b) => a.rank - b.rank);
  const groups = new Map(ranks.map(r => [r, hand.filter(c => c.rank === r)]));
  const add = cards => {
    const type = classify(cards, level);
    if (!beats(type, target)) return;
    const key = cards.map(c => c.id).sort((a, b) => a - b).join(',');
    found.set(key, { cards, type });
  };
  const take = (rank, n, used = []) => {
    const unavailable = new Set(used.map(c => c.id));
    const real = groups.get(rank).filter(c => !unavailable.has(c.id)).slice(0, n);
    const extra = wild.filter(c => !unavailable.has(c.id)).slice(0, n - real.length);
    return real.length + extra.length === n ? [...real, ...extra] : null;
  };
  hand.forEach(c => add([c]));
  for (const n of [2, 3, 5]) {
    ranks.forEach(r => { const cards = take(r, n); if (cards) add(cards); });
    if (wild.length >= n) {
      add(wild.slice(0, n));
      add(wild.slice(-n));
    }
  }
  for (const r of ranks) {
    const quad = take(r, 4);
    if (quad) {
      const kicker = [...hand].sort((a, b) => strength(a.rank, level) - strength(b.rank, level)).find(c => !quad.some(q => q.id === c.id));
      if (kicker) add([...quad, kicker]);
    }
    const triple = take(r, 3);
    if (triple) ranks.filter(s => s !== r).forEach(s => {
      const pair = take(s, 2, triple);
      if (pair) add([...triple, ...pair]);
    });
  }
  for (const suit of [-1, 0, 1, 2, 3]) {
    for (const w of windows) {
      const picked = [];
      let wi = 0;
      for (const r of w) {
        const card = groups.get(r).find(c => suit === -1 || c.suit === suit) ?? wild[wi++];
        if (!card) break;
        picked.push(card);
      }
      if (picked.length === 5) add(picked);
    }
    if (suit !== -1) {
      const suited = hand.filter(c => c.suit === suit).sort((a, b) => strength(a.rank, level) - strength(b.rank, level));
      for (let start = 0; start <= Math.max(0, suited.length - 5); start++) {
        const slice = suited.slice(start, start + 5);
        add([...slice, ...wild.slice(0, 5 - slice.length)]);
      }
    }
  }
  if (hand.length <= 5) add(hand);
  return [...found.values()].sort((a, b) => a.type.count - b.type.count || a.type.category - b.type.category || compare(a.type.values, b.type.values));
}

export function secureRandomIndex(limit) {
  if (!Number.isInteger(limit) || limit < 1 || limit > 2 ** 32) throw new Error('随机范围无效');
  const bound = Math.floor(2 ** 32 / limit) * limit;
  const buffer = new Uint32Array(1);
  do { globalThis.crypto.getRandomValues(buffer); } while (buffer[0] >= bound);
  return buffer[0] % limit;
}

export function newGame(previous = null, random = null) {
  const levels = previous ? [...previous.levels] : [2, 2];
  const dealer = previous?.dealer ?? 0;
  const level = levels[dealer];
  const deck = createDeck();
  for (let i = deck.length - 1; i > 0; i--) {
    const j = random ? Math.floor(random() * (i + 1)) : secureRandomIndex(i + 1);
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }
  const game = {
    version: 2, round: (previous?.round ?? 0) + 1, levels, dealer, level, direction: -1,
    hands: Array.from({ length: 6 }, (_, i) => sortHand(deck.slice(i * 27, (i + 1) * 27), level)),
    turn: previous?.order?.[0] ?? dealer, target: null, lastPlayer: null,
    passed: [], order: [], over: false, result: null, phase: 'playing',
    played: [], history: [], foundationHistory: [], questions: [], reports: Array(6).fill(null), exchanges: [], returnIndex: 0,
    lead: previous?.order?.[0] ?? dealer,
    logs: [`第 ${(previous?.round ?? 0) + 1} 局开桌，${dealer === 0 ? '青队' : '橙队'}坐庄，打 ${rankName(level)}。`],
    actions: Array(6).fill(null),
  };
  if (previous?.result?.winner !== null && previous?.result?.trapped?.length) {
    const winner = previous.result.winner;
    const receivers = previous.order.filter(i => i % 2 === winner).slice(0, previous.result.trapped.length);
    const donors = [...previous.result.trapped].sort((a, b) => seatDistance(game, game.lead, a) - seatDistance(game, game.lead, b));
    for (const from of donors) {
      receivers.sort((a, b) => seatDistance(game, from, a) - seatDistance(game, from, b));
      const to = receivers.shift();
      const card = sortHand(game.hands[from], level)[0];
      game.hands[from] = game.hands[from].filter(c => c.id !== card.id);
      game.hands[to].push(card);
      game.exchanges.push({ from, to, tribute: card, returned: null });
      log(game, `${NAMES[from]}向${NAMES[to]}进贡${rankName(card.rank)}。`);
    }
    game.phase = 'returning';
    game.turn = game.exchanges[0].to;
    log(game, '进贡完毕，受贡者依次手选一张还贡。');
  }
  return game;
}

function log(game, message) {
  game.logs.push(message);
  game.logs = game.logs.slice(-40);
}

export function publicPolicyHistory(game) {
  if (game.foundationHistory !== undefined) return game.foundationHistory;
  // Legacy saves can recover plays and questions, but not earlier passes.
  const events = [];
  let questionIndex = 0;
  let played = game.played.length - game.history.reduce((n, event) => n + event.cards.length, 0);
  for (const event of game.history) {
    while (questionIndex < game.questions.length && game.questions[questionIndex].playedAt <= played) {
      const question = game.questions[questionIndex++];
      events.push({ kind: 'ask', player: question.player, other: question.other,
        answer: question.answer, playedAt: question.playedAt, lead: question.target === null });
    }
    events.push({ kind: 'play', player: event.player, cards: event.cards, lead: event.lead });
    played += event.cards.length;
  }
  for (const question of game.questions.slice(questionIndex)) {
    events.push({ kind: 'ask', player: question.player, other: question.other,
      answer: question.answer, playedAt: question.playedAt, lead: question.target === null });
  }
  return events;
}

function recordPublicEvent(game, event) {
  if (game.foundationHistory === undefined) game.foundationHistory = publicPolicyHistory(game);
  game.foundationHistory.push(event);
}

function seatDistance(game, from, to) {
  return (game.direction === 1 ? to - from + 6 : from - to + 6) % 6;
}

function nextActive(game, from) {
  for (let n = 1; n <= 6; n++) {
    const i = (from + (game.direction ?? -1) * n + 6) % 6;
    if (game.hands[i].length) return i;
  }
  throw new Error('没有可行动的玩家');
}

function finish(game) {
  const completeTeam = [0, 1].find(team => game.order.filter(i => i % 2 === team).length === 3);
  if (completeTeam === undefined) return;
  const finishedCount = game.order.length;
  const trapped = game.hands.map((h, i) => h.length ? i : -1).filter(i => i >= 0);
  trapped.sort((a, b) => seatDistance(game, game.order[0], a) - seatDistance(game, game.order[0], b));
  game.order.push(...trapped);
  game.over = true;
  game.phase = 'over';
  const head = game.order[0] % 2;
  const winner = completeTeam === head ? head : null;
  const sweep = winner !== null && trapped.length === 3;
  let message;
  if (winner === game.dealer) {
    const old = game.levels[winner];
    game.levels[winner] = Math.min(14, old + 1);
    message = old === 13 ? '庄家打过 K，完成通关！' : `庄家升一级，下局打 ${rankName(game.levels[winner])}。`;
  } else if (head !== game.dealer) {
    game.dealer = head;
    message = `换庄，下局${head === 0 ? '青队' : '橙队'}打 ${rankName(game.levels[head])}。`;
  } else {
    message = '本局平局，庄家保级。';
  }
  const tributeMessage = winner !== null ? `${trapped.map(i => NAMES[i]).join('、')}下局进贡。` : '平局，不进贡。';
  game.result = { winner, sweep, message: `${message}${tributeMessage}`, champion: game.levels.some(l => l === 14), trapped, finishedCount };
  log(game, message);
}

export function playCards(game, player, ids) {
  if (game.phase !== 'playing' || game.over || game.turn !== player) throw new Error('还没轮到你出牌');
  if (!ids.length || new Set(ids).size !== ids.length) throw new Error('请先选择要出的牌');
  const cards = ids.map(id => game.hands[player].find(c => c.id === id));
  if (cards.some(c => !c)) throw new Error('所选牌不在手牌中');
  const type = classify(cards, game.level);
  if (!type) throw new Error('这不是有效牌型，只能出 1、2、3 或 5 张');
  if (!beats(type, game.target?.type)) throw new Error('需要同张数、比桌面更大的牌');
  game.hands[player] = game.hands[player].filter(c => !ids.includes(c.id));
  recordPublicEvent(game, { kind: 'play', player, cards, lead: game.target === null });
  game.played.push(...cards.map(card => ({ ...card, player })));
  game.history.push({ player, cards, type, lead: game.target === null });
  game.reports[player] = null;
  game.target = { cards, type };
  game.lastPlayer = player;
  game.passed = [];
  game.actions[player] = { text: type.name, cards };
  log(game, `${NAMES[player]}出了${type.name}（${cards.map(c => rankName(c.rank)).join(' ')}）。`);
  if (!game.hands[player].length) {
    game.order.push(player);
    log(game, `${NAMES[player]}出完，获得第 ${game.order.length} 名！`);
  }
  finish(game);
  if (!game.over) game.turn = nextActive(game, player);
}

export function passTurn(game, player) {
  if (game.phase !== 'playing' || game.over || game.turn !== player) throw new Error('还没轮到你');
  if (!game.target) throw new Error('领出时不能不出');
  recordPublicEvent(game, { kind: 'pass', player, cards: [], lead: false });
  game.passed.push(player);
  game.actions[player] = { text: '不出', cards: [] };
  log(game, `${NAMES[player]}不出。`);
  const remaining = game.hands.map((h, i) => h.length && i !== game.lastPlayer ? i : -1).filter(i => i !== -1);
  if (remaining.every(i => game.passed.includes(i))) {
    const owner = game.lastPlayer;
    if (game.nextDirection) {
      game.direction = game.nextDirection;
      delete game.nextDirection;
      log(game, '旧轮已结束，现在按逆时针出牌，下位玩家接风。');
    }
    game.turn = game.hands[owner].length ? owner : nextActive(game, owner);
    game.target = null;
    game.lastPlayer = null;
    game.passed = [];
    game.actions = Array(6).fill(null);
    log(game, `${NAMES[game.turn]}领出${game.hands[owner].length ? '' : '（接风）'}。`);
  } else {
    game.turn = nextActive(game, player);
  }
}

export function canAsk(game, player = game.turn, other = game.lastPlayer) {
  if (game.phase !== 'playing' || game.over || game.turn !== player) return false;
  if (!game.target) return other !== null && other !== player && other % 2 === player % 2 &&
    Number.isInteger(other) && other >= 0 && other < 6 && game.hands[other].length > 0;
  return other === game.lastPlayer && candidates(game.hands[player], game.level, game.target.type).length > 0;
}

export function askCards(game, player = game.turn, other = game.lastPlayer) {
  if (!canAsk(game, player, other)) throw new Error('跟牌时须能压过桌面并问上一出牌者；领出时只能问仍有牌的队友');
  const count = game.hands[other].length;
  const answer = count <= 10 ? count : null;
  recordPublicEvent(game, { kind: 'ask', player, other, answer, playedAt: game.played.length, lead: game.target === null });
  game.reports[other] = { answer, playedAt: game.played.length };
  game.questions.push({ player, other, answer, playedAt: game.played.length, target: game.target?.type ?? null });
  log(game, `${NAMES[player]}问${NAMES[other]}牌：${answer === null ? '不报（超过 10 张）' : `${answer} 张`}。`);
  return { player: other, count: answer };
}

export function returnCard(game, player, id) {
  const exchange = game.exchanges[game.returnIndex];
  if (game.phase !== 'returning' || player !== game.turn || player !== exchange?.to) throw new Error('还没轮到你还贡');
  const card = game.hands[player].find(c => c.id === id);
  if (!card) throw new Error('还贡牌必须在你的手牌中');
  if (card.id === exchange.tribute.id) throw new Error('不能把刚收到的贡牌原样退回');
  game.hands[player] = game.hands[player].filter(c => c.id !== id);
  game.hands[exchange.from].push(card);
  exchange.returned = card;
  log(game, `${NAMES[player]}已向${NAMES[exchange.from]}还贡一张。`);
  game.returnIndex++;
  if (game.returnIndex === game.exchanges.length) {
    game.phase = 'playing';
    game.turn = game.lead;
    game.hands = game.hands.map(h => sortHand(h, game.level));
    log(game, `还贡完成，${NAMES[game.lead]}先出。`);
  } else {
    game.turn = game.exchanges[game.returnIndex].to;
  }
}

const plannerCache = new Map();

function makePlanner(hand, level) {
  const key = `${level}:${hand.map(c => c.id).sort((a, b) => a - b).join(',')}`;
  if (plannerCache.has(key)) return plannerCache.get(key);
  const options = candidates(hand, level).map(move => {
    const jokers = move.cards.filter(c => c.rank > 14).length;
    const substitution = move.type.values[0] < 18;
    return {
      ...move,
      mask: move.cards.reduce((mask, c) => mask | (1 << hand.findIndex(h => h.id === c.id)), 0),
      cost: 1 + (move.type.count === 1 && move.type.values[0] < 14 ? .28 : 0) +
        (substitution ? jokers * .3 + (jokers >= 2 ? .55 : 0) : 0) -
        (move.type.count <= 3 && move.type.values[0] >= 18 || move.type.category >= 5 ? .08 : 0),
    };
  });
  const byCard = hand.map((_, i) => options.filter(o => o.mask & (1 << i)).sort((a, b) => b.cards.length - a.cards.length || a.cost - b.cost));
  const memo = new Map([[0, { score: 0, moves: [] }]]);
  let budget = 1800;
  function greedy(mask) {
    const moves = [];
    let score = 0;
    while (mask) {
      const move = options.filter(o => (o.mask & mask) === o.mask).sort((a, b) => a.cost / a.cards.length - b.cost / b.cards.length)[0];
      moves.push(move);
      score += move.cost;
      mask ^= move.mask;
    }
    return { score, moves };
  }
  function solve(mask) {
    if (memo.has(mask)) return memo.get(mask);
    if (budget-- <= 0) {
      const result = greedy(mask);
      memo.set(mask, result);
      return result;
    }
    let anchor = 0;
    let choices = null;
    for (let i = 0; i < hand.length; i++) {
      if (!(mask & (1 << i))) continue;
      const valid = byCard[i].filter(o => (o.mask & mask) === o.mask);
      if (choices === null || valid.length < choices.length) {
        anchor = i;
        choices = valid;
      }
    }
    let best = greedy(mask);
    for (const move of byCard[anchor].filter(o => (o.mask & mask) === o.mask).slice(0, 14)) {
      if (move.cost + Math.ceil((bitCount(mask) - move.cards.length) / 5) * .92 >= best.score) continue;
      const rest = solve(mask ^ move.mask);
      const score = move.cost + rest.score;
      if (score < best.score) best = { score, moves: [move, ...rest.moves] };
    }
    memo.set(mask, best);
    return best;
  }
  const full = (1 << hand.length) - 1;
  const root = solve(full);
  const planner = { options, full, root, solve };
  plannerCache.set(key, planner);
  if (plannerCache.size > 80) plannerCache.delete(plannerCache.keys().next().value);
  return planner;
}

function bitCount(mask) {
  let count = 0;
  while (mask) { mask &= mask - 1; count++; }
  return count;
}

export function planHand(hand, level = 2) {
  const result = makePlanner(hand, level).root;
  return { turns: result.moves.length, score: result.score, moves: result.moves.map(m => m.cards) };
}

export function publicCounts(game) {
  const counts = Array(6).fill(27);
  game.played.forEach(c => counts[c.player]--);
  return counts;
}

function sampledThreats(game, player) {
  // Hidden hands are sampled from unseen cards, never read from opponents.
  const own = game.hands[player];
  const seen = new Set([...own, ...game.played].map(c => c.id));
  const unseen = createDeck().filter(c => !seen.has(c.id));
  const counts = publicCounts(game);
  const enemies = [0, 1, 2, 3, 4, 5].filter(i => i % 2 !== player % 2 && counts[i] > 0);
  let seed = own.reduce((s, c) => (s * 31 + c.id) >>> 0, game.played.length + 19);
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const samples = [];
  for (let sample = 0; sample < 8; sample++) {
    const deck = [...unseen];
    for (let i = deck.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }
    let offset = 0;
    const threats = [];
    for (let i = 0; i < 6; i++) {
      if (i === player) continue;
      const hand = deck.slice(offset, offset + counts[i]);
      offset += counts[i];
      if (!enemies.includes(i)) continue;
      const best = new Map();
      for (const move of candidates(hand, game.level)) {
        if (beats(move.type, best.get(move.type.count))) best.set(move.type.count, move.type);
      }
      threats.push({ player: i, best });
    }
    samples.push(threats);
  }
  return { samples, counts, enemies };
}

function makeRoutes(game, player, planner, counts) {
  const hand = game.hands[player];
  const seen = new Set([...hand, ...game.played].map(c => c.id));
  const unseen = createDeck().filter(c => !seen.has(c.id));
  const enemyCount = Math.max(0, ...counts.filter((_, i) => i % 2 !== player % 2));
  const ceilings = new Map();
  // Pool all unseen cards: this overestimates enemy strength, without peeking.
  for (const move of candidates(unseen, game.level)) {
    if (move.type.count <= enemyCount && beats(move.type, ceilings.get(move.type.count))) {
      ceilings.set(move.type.count, move.type);
    }
  }
  const remainder = mask => classify(hand.filter((_, i) => mask & (1 << i)), game.level);
  const holds = move => !beats(ceilings.get(move.type.count), move.type);
  const recovers = move => !ceilings.has(move.type.count) || beats(move.type, ceilings.get(move.type.count));
  const closes = move => !!remainder(planner.full ^ move.mask);
  const stages = move => {
    const rest = planner.full ^ move.mask;
    return planner.options.some(control => control.type.count === move.type.count &&
      (control.mask & rest) === control.mask && recovers(control) && remainder(rest ^ control.mask));
  };
  const follows = (move, target) => {
    const road = move.type.count;
    const planned = planner.root.moves;
    return target && [1, 2, 3].includes(road) && target.values[0] < 14 && move.type.values[0] < 14 &&
      move.cards.every(c => c.rank < 19) && planned.length >= 2 &&
      planned.every(other => other.type.count === road && !beats(move.type, other.type)) &&
      planned.some(other => other.mask === move.mask) && planned.some(other => beats(other.type, move.type));
  };
  return { holds, closes, stages, follows };
}

export function suggestion(game, player = game.turn) {
  const hand = game.hands[player];
  if (game.phase === 'returning') {
    const tributeId = game.exchanges[game.returnIndex].tribute.id;
    const planner = makePlanner(hand, game.level);
    const choices = planner.options.filter(o => o.cards.length === 1 && o.cards[0].id !== tributeId);
    choices.sort((a, b) => {
      const cost = o => planner.solve(planner.full ^ o.mask).score * 8 + strength(o.cards[0].rank, game.level) / 3;
      return cost(a) - cost(b);
    });
    return choices[0]?.cards ?? null;
  }
  const planner = makePlanner(hand, game.level);
  let options = planner.options.filter(o => beats(o.type, game.target?.type));
  const finishMove = options.find(o => o.cards.length === hand.length);
  if (finishMove) return finishMove.cards;
  if (!options.length) return null;
  const counts = publicCounts(game);
  const allyLeading = game.target && game.lastPlayer % 2 === player % 2;
  const routes = makeRoutes(game, player, planner, counts);
  const exits = options.filter(move => routes.closes(move) && routes.holds(move));
  exits.sort((a, b) => planner.solve(planner.full ^ a.mask).score - planner.solve(planner.full ^ b.mask).score ||
    compare(a.type.values, b.type.values));
  if (exits.length) return exits[0].cards;
  let finishingEnemies = [];
  if (allyLeading) {
    const planned = new Set(planner.root.moves.map(move => move.mask));
    const urgent = counts.some((count, i) => i % 2 !== player % 2 && count === game.target.type.count);
    if (!urgent && hand.length <= 15) {
      const receiving = options.filter(move => move.cards.every(c => c.rank < 19) && routes.stages(move));
      receiving.sort((a, b) => planner.solve(planner.full ^ a.mask).score - planner.solve(planner.full ^ b.mask).score ||
        compare(a.type.values, b.type.values));
      if (receiving.length) return receiving[0].cards;
      const following = options.filter(move => routes.follows(move, game.target.type));
      following.sort((a, b) => compare(a.type.values, b.type.values));
      if (following.length) return following[0].cards;
    }
    let nextLeader = null;
    if (counts[game.lastPlayer] === 0) {
      for (let n = 1; n <= 6; n++) {
        const i = (game.lastPlayer + (game.direction ?? -1) * n + 6) % 6;
        if (counts[i] > 0) { nextLeader = i; break; }
      }
    }
    finishingEnemies = [0, 1, 2, 3, 4, 5].filter(i => i % 2 !== player % 2 &&
      counts[i] === game.target.type.count && (!game.passed.includes(i) || i === nextLeader));
    if (!finishingEnemies.length) return null;
    const covers = options.filter(move => (move.type.count === 1 || planned.has(move.mask)) &&
      (move.type.count === 1 || move.cards.every(c => c.rank < 19)));
    if (!covers.length) return null;
    const { samples } = sampledThreats(game, player);
    const winningRisk = type => samples.filter(sample => sample.some(t =>
      finishingEnemies.includes(t.player) && beats(t.best.get(type.count), type))).length / samples.length;
    const anyRisk = type => samples.filter(sample => sample.some(t => beats(t.best.get(type.count), type))).length / samples.length;
    if (winningRisk(game.target.type) < .75) return null;
    const safe = covers.filter(move => winningRisk(move.type) <= .125 && anyRisk(move.type) <= .25);
    safe.sort((a, b) => planner.solve(planner.full ^ a.mask).score - planner.solve(planner.full ^ b.mask).score ||
      compare(a.type.values, b.type.values));
    return safe[0]?.cards ?? null;
  }
  const { samples, enemies } = sampledThreats(game, player);
  const danger = enemies.some(i => counts[i] <= 5);
  const risks = new Map();
  const risk = type => {
    const key = `${type.count}:${type.category}:${type.values.join(',')}`;
    if (!risks.has(key)) risks.set(key,
      samples.filter(sample => sample.some(t => beats(t.best.get(type.count), type))).length / samples.length);
    return risks.get(key);
  };
  const urgent = enemies.some(i => counts[i] === 5);
  options = options.filter(move => !(move.type.count === 5 && move.type.values[0] < 18 &&
    move.cards.filter(c => c.rank > 14).length >= 2 && !routes.holds(move) && !urgent));
  if (!options.length) return null;
  const allies = [0, 1, 2, 3, 4, 5].filter(i => i !== player && i % 2 === player % 2 && counts[i] > 0);
  const publicRoads = allies.map(i => game.history.findLast(move => move.player === i))
    .filter(move => move?.lead && counts[move.player] >= move.cards.length).map(move => move.cards.length);
  const score = move => {
    const leftover = planner.solve(planner.full ^ move.mask);
    const exposure = risk(move.type);
    const control = exposure * (routes.closes(move) ? 7 : danger ? 5 : !game.target ? 1.2 : .65);
    const spent = move.cards.reduce((sum, c) => sum + (c.rank > 14 ? .55 : c.rank === game.level ? .23 : 0), 0);
    let support = 0;
    if (!game.target && publicRoads.includes(move.cards.length)) support -= .3;
    if (!game.target && danger && move.type.count === 1 && exposure > .5) support += 2;
    return leftover.score * 3 + control + spent + support + move.type.category * .04;
  };
  options.sort((a, b) => (planner.solve(planner.full ^ a.mask).score - planner.solve(planner.full ^ b.mask).score) ||
    a.type.category - b.type.category || compare(a.type.values, b.type.values));
  const shortlist = [...options.slice(0, 18), ...options.slice(-3)];
  shortlist.sort((a, b) => score(a) - score(b));
  return shortlist[0].cards;
}

export function computerTurn(game, policyIds = undefined, { automaticAsking = true } = {}) {
  if (policyIds !== undefined && (!Array.isArray(policyIds) || new Set(policyIds).size !== policyIds.length ||
    policyIds.some(id => !game.hands[game.turn].some(c => c.id === id)))) throw new Error('本地模型选择了不属于手牌的牌');
  if (game.phase === 'returning') {
    if (policyIds !== undefined && policyIds.length !== 1) throw new Error('本地模型还贡必须选择一张牌');
    const cards = policyIds === undefined ? suggestion(game) : game.hands[game.turn].filter(c => policyIds.includes(c.id));
    return returnCard(game, game.turn, cards[0].id);
  }
  const cards = policyIds === undefined ? suggestion(game)
    : policyIds.length ? game.hands[game.turn].filter(c => policyIds.includes(c.id)) : null;
  if (policyIds !== undefined && (cards ? !beats(classify(cards, game.level), game.target?.type) : !game.target)) {
    throw new Error('本地模型选择了不合法的出牌');
  }
  const counts = publicCounts(game);
  const player = game.turn;
  const allyLeading = game.target && game.lastPlayer % 2 === player % 2;
  let reason = null;
  if (cards && policyIds !== undefined) reason = '本地模型选择（试验）';
  else if (cards && allyLeading) {
    const hand = game.hands[player];
    const planner = makePlanner(hand, game.level);
    const mask = cards.reduce((mask, c) => mask | (1 << hand.findIndex(h => h.id === c.id)), 0);
    const routes = makeRoutes(game, player, planner, counts);
    if (cards.length === hand.length) reason = '一次出完，不留尾牌';
    else {
      const move = { cards, mask, type: classify(cards, game.level) };
      if (routes.closes(move) && routes.holds(move)) reason = '接同伴送牌，两手收完';
      else if (routes.stages(move)) reason = '接同伴送牌，留同路回手再收尾';
      else if (routes.follows(move, game.target.type)) {
        reason = `接同伴送牌，继续${({ 1: '单张', 2: '对子', 3: '三张' })[cards.length]}路`;
      } else reason = counts[game.lastPlayer] === 0 ? '残局守门，避免对手接风' : '残局守门，阻止对手一手出完';
    }
  }
  if (automaticAsking && cards && canAsk(game) && counts[game.lastPlayer] <= 10 &&
    game.reports[game.lastPlayer]?.answer !== counts[game.lastPlayer]) askCards(game);
  if (automaticAsking && !game.target) {
    for (const ally of [0, 1, 2, 3, 4, 5].filter(i => i !== game.turn && i % 2 === game.turn % 2)) {
      if (canAsk(game, game.turn, ally) && counts[ally] <= 10 &&
        game.reports[ally]?.answer !== counts[ally]) askCards(game, game.turn, ally);
    }
  }
  if (cards) {
    if (reason) log(game, `${NAMES[player]}${policyIds !== undefined ? '出牌' : '接手同伴'}：${reason}。`);
    playCards(game, player, cards.map(c => c.id));
    if (reason) {
      game.history.at(-1).reason = reason;
      game.actions[player].reason = reason;
    }
  }
  else passTurn(game, game.turn);
}

export function validSave(game) {
  if (!game || game.version !== 2 || !Array.isArray(game.hands) || game.hands.length !== 6 || !game.hands.every(Array.isArray) ||
    !Array.isArray(game.levels) || game.levels.length !== 2 || !game.levels.every(n => Number.isInteger(n) && n >= 2 && n <= 14) ||
    !Number.isInteger(game.turn) || game.turn < 0 || game.turn > 5 ||
    ![0, 1].includes(game.dealer) || game.level !== game.levels[game.dealer] && !game.over ||
    !Array.isArray(game.order) || !Array.isArray(game.passed) || !Array.isArray(game.logs) ||
    !Array.isArray(game.actions) || game.actions.length !== 6 || typeof game.over !== 'boolean' ||
    !['playing', 'returning', 'over'].includes(game.phase) ||
    !Array.isArray(game.played) || !Array.isArray(game.history) || !Array.isArray(game.questions) || !Array.isArray(game.reports) || game.reports.length !== 6 ||
    !Array.isArray(game.exchanges) || !Number.isInteger(game.returnIndex) || game.returnIndex < 0 || game.returnIndex > game.exchanges.length ||
    !Number.isInteger(game.round) || game.round < 1 || !Number.isInteger(game.level) || game.level < 2 || game.level > 13 ||
    !Number.isInteger(game.lead) || game.lead < 0 || game.lead > 5 ||
    game.direction !== undefined && ![-1, 1].includes(game.direction) ||
    game.nextDirection !== undefined && game.nextDirection !== -1 ||
    game.over !== (game.phase === 'over') ||
    new Set(game.order).size !== game.order.length || game.order.length > 6 ||
    !game.logs.every(s => typeof s === 'string') ||
    !game.reports.every(r => r === null || r && (r.answer === null || Number.isInteger(r.answer) && r.answer >= 0 && r.answer <= 10) && Number.isInteger(r.playedAt)) ||
    game.over && (!game.result || ![null, 0, 1].includes(game.result.winner) ||
      !Array.isArray(game.result.trapped) || !Number.isInteger(game.result.finishedCount) ||
      typeof game.result.message !== 'string' || typeof game.result.champion !== 'boolean')) return false;
  const deck = createDeck();
  const cards = [...game.hands.flat(), ...game.played];
  const validCard = c => c && Number.isInteger(c.id) && deck[c.id]?.rank === c.rank && deck[c.id]?.suit === c.suit;
  const seat = value => Number.isInteger(value) && value >= 0 && value < 6;
  if (!cards.every(validCard)) return false;
  if (!game.questions.every(q => q && seat(q.player) && seat(q.other) && q.player !== q.other &&
    (q.answer === null || Number.isInteger(q.answer) && q.answer >= 0 && q.answer <= 10) &&
    Number.isInteger(q.playedAt) && q.playedAt >= 0 && q.playedAt <= game.played.length)) return false;
  const publicCards = new Map(game.played.map(c => [c.id, c.player]));
  const questionKey = q => `${q.player}:${q.other}:${q.answer}:${q.playedAt}`;
  const questions = new Set(game.questions.map(questionKey));
  if (game.foundationHistory !== undefined && (!Array.isArray(game.foundationHistory) || game.foundationHistory.length > 4096 ||
    !game.foundationHistory.every(event => event && seat(event.player) && typeof event.lead === 'boolean' &&
      (event.kind === 'ask'
        ? seat(event.other) && (event.answer === null || Number.isInteger(event.answer) && event.answer >= 0 && event.answer <= 10) &&
          Number.isInteger(event.playedAt) && questions.has(questionKey(event)) && event.cards === undefined
        : ['play', 'pass'].includes(event.kind) && Array.isArray(event.cards) &&
          (event.kind === 'pass' ? event.cards.length === 0 && !event.lead : [1, 2, 3, 5].includes(event.cards.length)) &&
          event.cards.every(c => validCard(c) && publicCards.get(c.id) === event.player))))) return false;
  if (game.policyEvents !== undefined && (!Array.isArray(game.policyEvents) || game.policyEvents.length > 6 ||
    !game.policyEvents.every(event => event && Number.isInteger(event.player) && event.player >= 0 && event.player < 6 &&
      typeof event.lead === 'boolean' && (event.pass === undefined || typeof event.pass === 'boolean') &&
      Array.isArray(event.cards) && (event.pass ? event.cards.length === 0 : [1, 2, 3, 5].includes(event.cards.length)) &&
      event.cards.every(c => validCard(c) && game.played.some(played => played.id === c.id && played.player === event.player))))) return false;
  if (!game.actions.every(a => a === null || a && typeof a.text === 'string' && Array.isArray(a.cards) && a.cards.every(validCard)) ||
    !game.history.every(h => h && Number.isInteger(h.player) && h.player >= 0 && h.player < 6 &&
      Array.isArray(h.cards) && [1, 2, 3, 5].includes(h.cards.length) &&
      h.cards.every(c => validCard(c) && publicCards.get(c.id) === h.player) && typeof h.lead === 'boolean') ||
    !game.exchanges.every(e => e && Number.isInteger(e.from) && Number.isInteger(e.to) && e.from >= 0 && e.from < 6 &&
      e.to >= 0 && e.to < 6 && e.from % 2 !== e.to % 2 && validCard(e.tribute) && (e.returned === null || validCard(e.returned)))) return false;
  return cards.length === 162 && new Set(cards.map(c => c?.id)).size === cards.length &&
    game.order.every(i => Number.isInteger(i) && i >= 0 && i < 6) &&
    game.passed.every(i => Number.isInteger(i) && i >= 0 && i < 6) &&
    game.played.every(c => Number.isInteger(c.player) && c.player >= 0 && c.player < 6) &&
    (game.over || game.hands[game.turn].length > 0) &&
    (game.phase !== 'returning' || game.exchanges[game.returnIndex]?.to === game.turn) &&
    (!game.target || Number.isInteger(game.lastPlayer) && game.lastPlayer >= 0 && game.lastPlayer < 6 &&
      Array.isArray(game.target.cards) && game.target.cards.every(c => c && deck[c.id]?.rank === c.rank && deck[c.id]?.suit === c.suit) &&
      JSON.stringify(classify(game.target.cards, game.level)) === JSON.stringify(game.target.type));
}
