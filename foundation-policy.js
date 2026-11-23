import { canAsk, askCards, computerTurn, publicPolicyHistory } from './engine.js';
import { observeForPolicy, legalPolicyActions } from './policy.js';

export const ENCODING_VERSION = 2;
export const OBSERVATION_SIZE = 922;
export const ACTION_SIZE = 81;
export const HISTORY_SIZE = 71;
const kinds = ['play', 'pass', 'return', 'ask'];
const oneHot = (size, index) => Array.from({ length: size }, (_, i) => Number(i === index));

export function observeFoundation(game, player = game.turn) {
  return observeForPolicy(game, player);
}

export function legalFoundationActions(game, player = game.turn) {
  const actions = legalPolicyActions(game, player).map(action => ({
    ...action, features: [...action.features, ...oneHot(4, kinds.indexOf(action.kind)), ...Array(6).fill(0)],
  }));
  if (game.phase !== 'playing' || game.over) return actions;
  for (let other = 0; other < 6; other++) {
    if (!canAsk(game, player, other) ||
      game.questions.some(q => q.player === player && q.other === other && q.playedAt === game.played.length)) continue;
    actions.push({
      kind: 'ask', target: other, ids: [],
      features: [...Array(71).fill(0), ...oneHot(4, 3), ...oneHot(6, (other - player + 6) % 6)],
    });
  }
  return actions;
}

export function encodeFoundationHistory(game, player = game.turn, start = 0) {
  return publicPolicyHistory(game).slice(start).map(event => {
    const cards = Array(54).fill(0);
    for (const card of event.cards ?? []) {
      const type = card.rank > 14 ? card.rank === 19 ? 52 : 53 : card.suit * 13 + card.rank - 2;
      cards[type] += 1 / 3;
    }
    return [
      ...cards, ...oneHot(6, (event.player - player + 6) % 6),
      ...oneHot(6, event.kind === 'ask' ? (event.other - player + 6) % 6 : -1),
      ...oneHot(3, ['play', 'pass', 'ask'].indexOf(event.kind)),
      event.kind === 'ask' ? event.answer === null ? -1 : event.answer / 10 : 0,
      Number(event.lead),
    ];
  });
}

export function applyFoundationAction(game, action) {
  const legal = action && Array.isArray(action.ids) && legalFoundationActions(game).find(candidate => candidate.kind === action.kind &&
    candidate.target === action.target && candidate.ids.length === action.ids.length &&
    candidate.ids.every((id, i) => id === action.ids[i]));
  if (!legal) throw new Error('本地模型选择了不合法的动作');
  if (legal.kind === 'ask') askCards(game, game.turn, legal.target);
  else computerTurn(game, legal.ids, { automaticAsking: false });
}
